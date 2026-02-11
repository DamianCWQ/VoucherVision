import time, torch, json, os
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_community.llms import Ollama
from langchain_community.chat_models import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import JsonOutputParser
from langchain_classic.output_parsers import RetryWithErrorOutputParser

# Load environment variables from .env file
load_dotenv()

from vouchervision.utils_LLM import SystemLoadMonitor, run_tools, count_tokens, save_individual_prompt, sanitize_prompt
from vouchervision.utils_LLM_JSON_validation import validate_and_align_JSON_keys_with_template

class OllamaHandler: 
    RETRY_DELAY = 5  # Wait 5 seconds before retrying
    MAX_RETRIES = 3  # Maximum number of retries
    STARTING_TEMP = 0.5
    TOKENIZER_NAME = 'gpt-4'  # Use GPT-4 for token counting approximation
    VENDOR = 'ollama'

    def __init__(self, cfg, logger, model_name, JSON_dict_structure, config_vals_for_permutation):
        self.cfg = cfg
        self.tool_WFO = self.cfg['leafmachine']['project']['tool_WFO']
        self.tool_GEO = self.cfg['leafmachine']['project']['tool_GEO']
        self.tool_wikipedia = self.cfg['leafmachine']['project']['tool_wikipedia']

        self.logger = logger
        self.model_name = model_name
        self.JSON_dict_structure = JSON_dict_structure
        
        self.monitor = SystemLoadMonitor(logger)
        self.has_GPU = torch.cuda.is_available() 

        # Ollama connection settings
        self.base_url = os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')
        
        # Get custom model name if using Ollama Custom
        self.custom_model_name = os.getenv('OLLAMA_CUSTOM_MODEL_NAME', None)
        
        ### Config
        self.config_vals_for_permutation = config_vals_for_permutation
        
        # Set up a parser
        self.parser = JsonOutputParser()

        self.prompt = PromptTemplate(
            template="Answer the user query.\n{format_instructions}\n{query}\n",
            input_variables=["query"],
            partial_variables={"format_instructions": self.parser.get_format_instructions()},
        )
        self._set_config()

    def _set_config(self):
        if self.config_vals_for_permutation:
            self.starting_temp = float(self.config_vals_for_permutation.get('ollama', {}).get('temperature', self.STARTING_TEMP))
            self.model_kwargs = {
                    'num_predict': self.config_vals_for_permutation.get('ollama', {}).get('max_tokens', 1024),
                    'temperature': self.starting_temp,
                    'top_p': self.config_vals_for_permutation.get('ollama', {}).get('top_p', 1.0),
                    'top_k': self.config_vals_for_permutation.get('ollama', {}).get('top_k', 40),
                    }
        else:
            self.starting_temp = float(self.STARTING_TEMP)
            self.model_kwargs = {
                    'num_predict': 1024,  # max tokens to generate
                    'temperature': self.starting_temp,
                    'top_p': 1.0,
                    'top_k': 40,
                    }
        
        self.temp_increment = float(0.2)
        self.adjust_temp = self.starting_temp 

        self._build_model_chain_parser()

    def _adjust_config(self):
        new_temp = self.adjust_temp + self.temp_increment
        if hasattr(self, 'json_report') and self.json_report:            
            self.json_report.set_text(text_main=f'Incrementing temperature from {self.adjust_temp} to {new_temp}')
        self.logger.info(f'Incrementing temperature from {self.adjust_temp} to {new_temp}')
        self.adjust_temp += self.temp_increment
        self.model_kwargs['temperature'] = self.adjust_temp   

    def _reset_config(self):
        if hasattr(self, 'json_report') and self.json_report:            
            self.json_report.set_text(text_main=f'Resetting temperature from {self.adjust_temp} to {self.starting_temp}')
        self.logger.info(f'Resetting temperature from {self.adjust_temp} to {self.starting_temp}')
        self.adjust_temp = self.starting_temp
        self.model_kwargs['temperature'] = self.starting_temp   
        
    def _build_model_chain_parser(self):
        # Determine which model to use
        # If using "Ollama Custom", use the custom model name from env var
        if self.model_name.lower() == 'custom' and self.custom_model_name:
            actual_model_name = self.custom_model_name
        else:
            actual_model_name = self.model_name
        
        # Store for later use
        self.actual_model_name = actual_model_name
            
        self.logger.info(f'Using Ollama model: {actual_model_name} at {self.base_url}')
        
        # Check if this is a vision model
        self.is_vision_model = 'vision' in actual_model_name.lower()
        
        # Use ChatOllama for better compatibility
        llm_to_use = ChatOllama(
            model=actual_model_name,
            base_url=self.base_url,
            temperature=self.model_kwargs.get('temperature'),
            top_p=self.model_kwargs.get('top_p'),
            top_k=self.model_kwargs.get('top_k'),
            num_predict=self.model_kwargs.get('num_predict'),
            format='json',  # Request JSON output
        )
        
        # Set up the retry parser
        self.retry_parser = RetryWithErrorOutputParser.from_llm(
            parser=self.parser,
            llm=llm_to_use,
            max_retries=self.MAX_RETRIES
        )
        
        # Prepare the chain
        self.chain = self.prompt | llm_to_use

    def call_llm_api_Ollama(self, prompt_template, json_report, paths):
        _____, ____, _, __, ___, json_file_path_wiki, txt_file_path_ind_prompt = paths
        self.json_report = json_report
        if self.json_report:            
            self.json_report.set_text(text_main=f'Sending request to Ollama ({self.model_name})')
        self.monitor.start_monitoring_usage()
        nt_in = 0
        nt_out = 0
        
        ind = 0
        while ind < self.MAX_RETRIES:
            ind += 1
            try:
                self.logger.info(f"Ollama config: {self.model_kwargs}")
                # Invoke the chain to generate prompt text
                response = self.chain.invoke(input={"query": prompt_template})

                response_text = response.content if hasattr(response, 'content') else str(response)

                # Use retry_parser to parse the response with retry logic
                try:
                    output = self.retry_parser.parse_with_prompt(response_text, prompt_value=prompt_template)
                except:
                    try:
                        output = json.loads(response_text)
                    except Exception as json_error:
                        self.logger.error(f'Failed to parse JSON: {json_error}')
                        self.logger.error(f'LLM call failed with response:\n{response_text}')
                        self._adjust_config()
                        self._build_model_chain_parser()
                        continue

                # Validate and align the output
                output = validate_and_align_JSON_keys_with_template(output, self.JSON_dict_structure)
                if output is None:
                    self.logger.error(f'[Attempt {ind}] Failed to extract valid JSON')
                    self._adjust_config()
                    self._build_model_chain_parser()
                    continue

                # Count tokens (approximate)
                nt_in = count_tokens(prompt_template, self.VENDOR, self.TOKENIZER_NAME)
                nt_out = count_tokens(response_text, self.VENDOR, self.TOKENIZER_NAME)

                # Stop monitoring and run tools
                self.monitor.stop_inference_timer()

                if self.json_report:
                    self.json_report.set_text(text_main=f'Working on WFO, Geolocation, Links')
                    
                output_WFO, WFO_record, output_GEO, GEO_record = run_tools(
                    output, self.tool_WFO, self.tool_GEO, self.tool_wikipedia, json_file_path_wiki
                )

                save_individual_prompt(sanitize_prompt(prompt_template), txt_file_path_ind_prompt)

                self.logger.info(f"Formatted JSON:\n{json.dumps(output, indent=4)}")

                usage_report = self.monitor.stop_monitoring_report_usage()

                if self.adjust_temp != self.starting_temp:
                    self._reset_config()

                if self.json_report:
                    self.json_report.set_text(text_main=f'LLM call successful')
                    
                return output, nt_in, nt_out, WFO_record, GEO_record, usage_report

            except Exception as e:
                self.logger.error(f'Attempt {ind} failed with error: {e}')
                self._adjust_config()
                self._build_model_chain_parser()
                time.sleep(self.RETRY_DELAY)

        # All retries failed
        self.logger.info(f"Failed to extract valid JSON after {self.MAX_RETRIES} attempts")
        if self.json_report:
            self.json_report.set_text(text_main=f'Failed to extract valid JSON after {self.MAX_RETRIES} attempts')

        self.monitor.stop_inference_timer()
        usage_report = self.monitor.stop_monitoring_report_usage()
        
        if self.json_report:
            self.json_report.set_text(text_main=f'LLM call failed')

        self._reset_config()
        return None, nt_in, nt_out, None, None, usage_report

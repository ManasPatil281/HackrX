import asyncio


class BatchProcessor:
    """Handles parallel processing of questions with rate limiting"""
    
    def __init__(self, llm, prompt_template, max_batch_size=3):
        self.llm = llm
        self.prompt_template = prompt_template
        self.max_batch_size = max_batch_size
        self.semaphore = asyncio.Semaphore(max_batch_size)
    
    async def process_question(self, question, context):
        """Process a single question with context using the LLM"""
        try:
            # Format prompt with question and context
            formatted_prompt = self.prompt_template.format(
                question=question,
                context=context
            )
            
            # Use LLM to generate answer
            if hasattr(self.llm, 'invoke_with_fallback'):
                response = await self.llm.invoke_with_fallback(formatted_prompt)
            else:
                response = self.llm.invoke(formatted_prompt)
            
            # Extract answer from response
            if hasattr(response, 'content'):
                answer = response.content
            else:
                answer = str(response)
            
            return answer
        except Exception as e:
            raise Exception(f"Error in batch processor: {str(e)}")

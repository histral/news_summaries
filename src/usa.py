from histral_core.firebase import Category
from src.common import save_summarization_output, get_news_list, call_gemini_api, Logger


# --------------------- Constants ---------------------


INSTRUCTIONS_PATH = "./instructions/usa.txt"
CATEGORY = Category.USA


# --------------------- Main Execution ---------------------


try:
    news_list = get_news_list(CATEGORY)

    response = call_gemini_api(news_list, INSTRUCTIONS_PATH)

    save_summarization_output(response, CATEGORY)
except Exception as e:
    Logger.error(f"FATAL: Error occurred in main execution; {e}")

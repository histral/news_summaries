import os
from unicodedata import category
import pytz
import logging as Logger
import google.generativeai as genai

from datetime import datetime
from histral_core.encode import decode_text
from histral_core.firebase import fetch_news_list, Category


# --------------------- Logging Setup ---------------------


Logger.basicConfig(
    level=Logger.INFO,
    format="[%(levelname)s] (%(asctime)s) -> %(message)s",
    handlers=[
        Logger.StreamHandler(),
    ],
)

# --------------------- Constants ---------------------


IST = pytz.timezone("Asia/Kolkata")
CURRENT_TIME_IST = datetime.now(IST)
CURRENT_DATE = CURRENT_TIME_IST.date()
API_KEY = os.getenv("API_KEY")


# --------------------- Common Functions ---------------------


def get_instructions(FILE_PATH):
    try:
        with open(FILE_PATH, "r") as file:
            data = file.read()

        return data
    except Exception as e:
        Logger.error(f"FATAL: Unable to read instructions from {FILE_PATH}")


def call_gemini_api(DATA: list, INSTRUCTIONS_PATH: str):
    try:
        genai.configure(api_key=API_KEY)

        generation_config = {
            "temperature": 1,
            "top_p": 0.95,
            "top_k": 64,
            "max_output_tokens": 8192,
            "response_mime_type": "text/plain",
        }

        model = genai.GenerativeModel(
            model_name="gemini-1.5-flash",
            generation_config=generation_config,
            system_instruction=get_instructions(INSTRUCTIONS_PATH),
        )

        chat_session = model.start_chat(history=[])

        response = chat_session.send_message(
            f"Fallowing is the list of news objects for today in 'India' category, collected from different outlets; ```{DATA}```"
        )

        if response._done != True:
            Logger.error("FATAL: Unknown error occurred while LLM API Call; {response}")
            raise ValueError(f"Unknown error occurred; {response}")

        return response
    except genai.CompletionResponseError as e:
        Logger.error(f"FATAL: Gemini API call failed with error: {e}")
        raise
    except Exception as e:
        Logger.error("FATAL: Unknown error occurred while LLM API Call; {e}")
        raise


def get_news_list(category: Category):
    try:
        news_list = []
        news_data = fetch_news_list(CURRENT_DATE, category=category)

        for key in news_data:
            if news_data.get(key) == None or len(news_data[key]) == 0:
                Logger.error(f"ERROR: No entries found for OutletCode {key}")
                continue

            for news in news_data[key]:
                if news.get("body") == None:
                    Logger.error(
                        f"ERROR: No body found for news in {key} w/ title ({news})"
                    )
                    continue

                news["body"] = decode_text(news["body"])
                news_list.append(news)

        Logger.info(
            f"INFO: Found total {len(news_list)} news objects for {category.value}"
        )

        return news_list
    except Exception as e:
        Logger.error("FATAL: Error occurred while fetching news data; {e}")
        raise


def save_summarization_output(response, category: Category, news_used: int):
    try:
        today_date = f"{CURRENT_DATE.day}-{CURRENT_DATE.month}-{CURRENT_DATE.year}"

        response_data = f"""
id: "{category.value}"
date: "{today_date}"
prompt_token: "{response.usage_metadata.prompt_token_count}"
response_token: "{response.usage_metadata.candidates_token_count}"
news_used: {news_used}
------
{response.text}
"""

        output_file_name = f"./summaries/{category.value}/{today_date}.md"

        with open(output_file_name, "w") as file:
            file.write(response_data)

        Logger.info(f"TRACE: Summery successfully saved in {output_file_name}")
    except Exception as e:
        Logger.error("FATAL: Unable to save output summery; {e}")
        raise

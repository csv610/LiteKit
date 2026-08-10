from litellm import completion
import argparse

def search_latest_news(topic, mode="summary", model="gemini/gemini-2.5-flash", context_size="high"):
    if mode == "list":
        prompt = f"What are some important news stories from today on the topic {topic}?"
    else:
        prompt = f"What was an important news story from today on the topic {topic}?"

    response = completion(
        model=model,
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        web_search_options={
            "search_context_size": context_size  # Options: "low", "medium", "high"
        }
    )
    return response.choices[0].message.content

def main():
    parser = argparse.ArgumentParser(description="Search for the latest news on a topic.")
    parser.add_argument("-i", "--topic", help="The topic to search for.")
    parser.add_argument("-m", "--mode", default="summary", choices=["summary", "list"], help="The mode of search.")
    parser.add_argument("-M", "--model", default="gemini-2.5-flash", help="The model to use for completion.")
    parser.add_argument("-c", "--context_size", default="medium", choices=["low", "medium", "high"], help="The search context size.")
    args = parser.parse_args()

    news  = search_latest_news(args.topic, args.mode, args.model, args.context_size)
    print(news)

if __name__ == "__main__":
    main()

import pandas as pd
import ollama
from tqdm import tqdm

#df = pd.read_csv("/mnt/data/wenlu/LLM_sentiment_affiliation/sentiment_analysis/data_Twitter/Twitter[topic=climatechange][model:llama+gemma]_[task:response+expand].csv")
#df.info()
sampled_df = pd.read_csv("claude/[Twitter]_claude_expanded_texts_v2.csv")
#sampled_df = df
sampled_df = sampled_df[sampled_df['claude_expanded_text'].isna()]      
sampled_df = sampled_df.drop(columns=['claude_expanded_text','expanded_text'])
sampled_df.info()
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_random_exponential

client = OpenAI(
    api_key="xxxxxxxxxxxxx", 
    base_url="xxxxxxxxxxxx",
)

sampled_df['claude_expanded_text'] = ''

from tenacity import retry, stop_after_attempt, wait_random_exponential, RetryError

@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def completion_with_backoff(**kwargs):
    return client.chat.completions.create(**kwargs)

failed_rows = []

for index, row in tqdm(sampled_df.iterrows(), total=sampled_df.shape[0], desc="Processing texts"):
    if row['claude_expanded_text']:
        continue

    try:
        completion = completion_with_backoff(
            model="claude-3-5-haiku-20241022-X", # model setting
            messages=[
                {"role": "system", "content": "..."},
                {"role": "user", "content": row['text']}
            ],
            temperature=0,
        )
        expanded_text = completion.choices[0].message.content
        sampled_df.at[index, 'claude_expanded_text'] = expanded_text

    except RetryError as e:
        print(f"RetryError on row {index}: {e}")
        if e.last_attempt:
            print(f"Last attempt failed with: {e.last_attempt.exception()}")
        sampled_df.at[index, 'claude_expanded_text'] = ''
        failed_rows.append(index)

    except Exception as e:
        print(f"Unhandled error on row {index}: {e}")
        sampled_df.at[index, 'claude_expanded_text'] = ''
        failed_rows.append(index)

# save data
sampled_df.to_csv("claude/[Twitter]_claude_expanded_texts_v3.csv", index=False)
print(f"Done. {len(failed_rows)} rows failed.")

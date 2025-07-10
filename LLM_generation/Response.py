# # 2. LLM Prediction
import pandas as pd
import ollama
from tqdm import tqdm

df = pd.read_csv("/mnt/data/wenlu/LLM_sentiment_affiliation/sentiment_analysis/data_Twitter/Twitter[topic=climatechange][model:llama+gemma]_[task:response+expand].csv")
df.info()
#sampled_df = pd.read_csv("claude/response_texts[Twitter].csv")
#sampled_df = sampled_df[sampled_df['claude_expanded_text'].isna()]
#sampled_df = sampled_df.drop(columns=['claude_expanded_text','expanded_text'])
#sampled_df.info()
sampled_df = df
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_random_exponential

client = OpenAI(
    api_key="xxxxxxxxxx", 
    base_url="xxxxx",
)

sampled_df['4o_response_text'] = ''

# 
@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def completion_with_backoff(**kwargs):
    return client.chat.completions.create(**kwargs)

# process text
for index, row in tqdm(sampled_df.iterrows(), total=sampled_df.shape[0], desc="Processing texts"):
    if row['4o_response_text']:  
        original_text = row['text']
    
    try:
        completion = completion_with_backoff(
            model="gpt-4o-all",  
            messages=[
                {"role": "user", "content": original_text} 
            ],
            temperature=0,
        )
        
        expanded_text = completion.choices[0].message.content
        sampled_df.at[index, '4o_response_text'] = expanded_text  # update the DataFrame
        
        sampled_df.to_csv("claude/response_texts[Twitter]_4o.csv", index=False)

        # time.sleep(1)
    
    except Exception as e:
        print(f"An error occurred on row {index}: {e}")
        sampled_df.at[index, '4o_response_text'] = ''  # 出错时添加空值 add empty value if something went wrong
sampled_df.to_csv("claude/response_texts[Twitter]_4o.csv", index=False)

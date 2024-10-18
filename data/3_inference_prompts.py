from http import HTTPStatus
import dashscope
import pandas as pd
import json

dashscope.api_key = "your api key"
def call_with_prompt(prompt_text):
    response = dashscope.Generation.call(
        model="qwen-turbo",
        prompt=prompt_text
    )
    if response.status_code == HTTPStatus.OK:
        print(response.output)
        return response.output.text
    else:
        print(response.code)
        print(response.message)
        return "None"


if __name__ == '__main__':
    dataset = "Beauty" # Office_Products | Grocery
    file_name = "train"
    prompts_csv = pd.read_csv(dataset + '/Prompts/typical_texts_all_{}_5.csv'.format(file_name))
    responses = []
    for index, row in prompts_csv.iterrows():
        prompt_text = row['Typical Text']
        response = call_with_prompt(prompt_text)
        responses.append(response)
    prompts_csv.loc[prompts_csv.index, 'Response'] = responses

    output_path = dataset + '/Prompts/typical_texts_with_responses_{}_5.csv'.format(file_name)
    prompts_csv.to_csv(output_path, index=False)





import requests
import csv
import os
import openai
from langchain.chains.llm import LLMChain

# https://medium.com/@akshathaholla91/templatized-email-body-generation-using-large-language-models-9e7aa3df0ad



customers_url = "https://media.githubusercontent.com/media/datablist/sample-csv-files/main/files/people/people-100.csv"
customers_file_name = "./customers.csv"

result=requests.get(customers_url)

with open(customers_file_name, "w") as file:

    file.write(result.text)

products_url = "https://diviengine.com/downloads/Divi-Engine-WooCommerce-Sample-Products.csv"
products_file_name = "./products.csv"
result=requests.get(products_url)
with open(products_file_name, "w") as file:
    file.write(result.text)

def read_csv(filename):
    with open(filename, 'r') as file:
        return list(csv.DictReader(file))

products = read_csv(products_file_name)
customers = read_csv(customers_file_name)

def generate_email_prompt(info: dict):
    prompt = """ Your goal is to write a personalized outbound email from {sales_rep}, a sales rep at  {company} to {prospect}.
    A good email is personalized and informs the customer on how your product can help the customer.
    Be sure to use value selling: A sales methodology that focuses on how your product or service will provide value to the customer instead of focusing on price or solution.
    % INFORMATION ABOUT {company}:
    a fashion and apparel company
    products list:
    {products_info}

    % INFORMATION ABOUT {prospect}:
    {text}

    % INCLUDE THE FOLLOWING PIECES IN YOUR RESPONSE:
    - Start the email with the sentence: "We have amazing discounts on our premium products such as ..." then insert the description of the product.
    - The sentence: "We observed that XYZ product suits your taste " Replace XYZ with one of the products
    - A 1-2 sentence description about the products, be brief
    - End your email with a call-to-action such as asking them to set up time to talk more

    % YOUR RESPONSE:"""
    for key in info.keys():
        prompt=prompt.replace("{"+key+"}",info[key])
    return prompt

customer_infos = []
for customer in customers[:5]:
    info = {}
    info['company'] = "XYZ"
    info['sales_rep'] = "ABC"
    info['prospect'] = customer["First Name"] + " " + customer["Last Name"]
    info['text'] = customer["Job Title"]
    products_info= ""
    i=1
    for product in products[:5]:
        products_info += f"\n {i}. Product Name: {product['Name']} \n  Description: {product['Short description']}"
        i+=1
    info['products_info']=products_info
    customer_infos.append(info)

llm_chain=LLMChain(prompt=prompt_template,llm=llm, verbose = True)


prompts = [generate_email_prompt(info) for info in customer_infos]

openai.api_key = os.getenv("OPENAI_API_KEY")

responses = [openai.Completion.create(
  model="text-davinci-003",
  prompt=prompt,
  temperature=0,
  max_tokens=3000,
  top_p=1.0,
  frequency_penalty=0.0,
  presence_penalty=0.0
) for prompt in prompts]

email_body = responses[0]["choices"][0].text

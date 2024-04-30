import streamlit as st
from transformers import pipeline
import torch
import csv
import re
import warnings

warnings.filterwarnings("ignore")

# Define a prompt template for Magicoder with placeholders for instruction and response.
MAGICODER_PROMPT = """You are an exceptionally intelligent coding assistant that consistently delivers accurate and reliable responses to user instructions.
@@ Instruction
{instruction}
@@ Response
"""

# Create a text generation pipeline using the Magicoder model, text-generation task, bfloat16 torch data type and auto device mapping.
generator = pipeline(
    model="ise-uiuc/Magicoder-S-DS-6.7B",
    task="text-generation",
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

# Function to generate response
def generate_response(instruction):
    prompt = MAGICODER_PROMPT.format(instruction=instruction)
    result = generator(prompt, max_length=2048, num_return_sequences=1, temperature=0.0)
    response = result[0]["generated_text"]
    response_start_index = response.find("@@ Response") + len("@@ Response")
    response = response[response_start_index:].strip()
    return response

# Function to append data to a CSV file
def save_to_csv(data, filename):
    with open(filename, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(data)

# Function to process user feedback
def process_output(correct_output):
    if correct_output.lower() == 'yes':
        feedback = st.text_input("Do you want to provide any feedback?")
        save_to_csv(["Correct", feedback], 'output_ratings.csv')
    else:
        correct_code = st.text_area("Please enter the correct code:")
        feedback = st.text_input("Any other feedback you want to provide:")
        save_to_csv(["Incorrect", feedback, correct_code], 'output_ratings.csv')

# Streamlit app
def main():
    st.title("Magicoder Assistant")

    instruction = st.text_area("Enter your instruction here:")
    if st.button("Generate Response"):
        generated_response = generate_response(instruction)
        st.text("Generated response:")
        st.text(generated_response)

        correct_output = st.radio("Is the generated output correct?", ("Yes", "No"))
        process_output(correct_output)

if __name__ == "__main__":
    main()

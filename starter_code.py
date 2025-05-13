# find_similar_talks_with_chatgpt.py
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import openai
import json
import tiktoken

# Load config
with open('config.json') as f:
    config = json.load(f)
    OPENAI_API_KEY = config['OPENAI_API_KEY']

# Initialize OpenAI client
client = openai.OpenAI(api_key=OPENAI_API_KEY)

def truncate_text(text, max_tokens, encoding_name="cl100k_base"):
    """Truncate text to fit within max_tokens using tiktoken."""
    encoding = tiktoken.get_encoding(encoding_name)
    tokens = encoding.encode(text)
    if len(tokens) > max_tokens:
        tokens = tokens[:max_tokens]
        text = encoding.decode(tokens)
    return text

def generate_chatgpt_response(search_term, talks, model="gpt-4o", max_context_tokens=3000):
    """
    Generate a ChatGPT response based on the search term and provided talks.
    Ensure that the response uses only the talk content, with no external sources.
    """
    try:
        # Return ChatGPT response based on the talks
        prompt = f"You are a helpful assistant that can answer questions about the following LDS leaders' talks/addresses: {talks}. The search term is: {search_term}. Please provide a few direct quotes from each talk, with no external sources, that relates to {search_term}"
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            stream=True
        )

        responseList = []
        for chunk in response:
            if chunk.choices[0].delta.content is not None:
                responseList.append(chunk.choices[0].delta.content)

        return "".join(responseList)
    
    except Exception as e:
        print(f"Error generating ChatGPT response: {e}")
        return "Unable to generate response due to an error."

def find_similar_talks(search_term, top_k=3):
    """
    Finds the top_k most similar conference talks to a given search term based on embeddings
    and generates a ChatGPT response based on those talks.
    """
    try:
        # Load the CSV
        df = pd.read_csv("cleaned_conference_talks.csv")

        # Initialize the sentence transformer model
        model = SentenceTransformer('all-mpnet-base-v2')

        # Generate embedding for the search term
        search_embedding = model.encode(search_term)
        """
        Pseudo code:
        Generate the search embedding from the search term using the model.
        For each row in the dataframe, you should extract the talk embedding and convert it
        from a string to a list (use eval()). Next, use the util.cos_sim() function
        to calculate the cosine similarity between the search embedding and the talk
        embedding. Save each similarity value for use the next step
        """

        # Find the top_k most similar talks and store them in a list.
        
        # Convert string embeddings to lists and calculate similarities
        similarities = []
        for index, row in df.iterrows():
            talk_embedding = eval(row['embeddings'])
            similarity_val = util.cos_sim(search_embedding, talk_embedding)[0][0].item()
            similarities.append(similarity_val)

        # Add similarities to dataframe and sort by similarity in descending order
        df['similarity'] = similarities
        df_sorted = df.sort_values('similarity', ascending=False)
        
        # Get top_k talks and prepare them for ChatGPT
        top_talks = df_sorted.head(top_k).to_dict('records')
        top_talks_prompts = []
        
        for talk_dict in top_talks:
            top_talk_content = talk_dict['title'] + " " + talk_dict['speaker'] + " " + str(talk_dict['year']) + " " + talk_dict['talk']
            top_talks_prompts.append(top_talk_content)

        # Generate ChatGPT response
        chatgpt_response = generate_chatgpt_response(search_term, top_talks_prompts)

        return top_talks, chatgpt_response

    except FileNotFoundError:
        print("Error: 'cleaned_conference_talks.csv' not found.")
        return [], ""
    except Exception as e:
        print(f"An error occurred: {e}")
        return [], ""

# Example usage
if __name__ == "__main__":
    search_term = "How can I deal with serious depression"
    similar_talks, chatgpt_response = find_similar_talks(search_term)

    if similar_talks:
        print(f"Top 3 most similar talks to '{search_term}':")
        for talk in similar_talks:
            print(
                f"- Title: {talk['title']}, "
                f"Speaker: {talk['speaker']}, "
                f"Year: {talk['year']}, "
                f"Similarity: {talk['similarity']:.4f}, "
                f"Text Snippet: {talk['talk'][:200].encode('ascii', 'ignore').decode()}"
            )
        print("\nChatGPT Response:")
        print(chatgpt_response)
    else:
        print("No similar talks found.")
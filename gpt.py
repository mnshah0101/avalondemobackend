from openai import OpenAI
import os

client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
)

# take in docs and create summary for each doc


def getDocSummary(docs):
    doc_summaries = {}
    for doc, name in docs:
        text = doc[0].page_content.split()
        text = " ".join(text[:min(len(text), 500)])

        prompt = f"Summarize the following text in 10-20 words, should be readable:\n\n{text}\n\n Only respond directly with the summary and nothing else"

        response = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model="gpt-3.5-turbo",
        )

        response = response.choices[0].message.content
        print(response)
        doc_summaries[name] = response

    return doc_summaries

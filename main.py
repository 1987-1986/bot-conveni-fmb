from keep_alive import keep_alive
import os
import logging
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, ContextTypes, filters
from openai import OpenAI
import faiss
import numpy as np
import pickle

keep_alive()

# Carreguem els fitxers pkl
with open("index.pkl", "rb") as f:
    index = pickle.load(f)

with open("chunks.pkl", "rb") as f:
    chunk_texts = pickle.load(f)

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Hola! Sóc el bot del conveni. Escriu la teva consulta!")

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    pregunta = update.message.text
    embedding = client.embeddings.create(
        input=pregunta,
        model="text-embedding-3-small"
    ).data[0].embedding

    D, I = index.search(np.array([embedding]).astype("float32"), 3)
    context_str = "\n---\n".join([chunk_texts[i] for i in I[0]])

    resposta = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "Respon només segons el context. Si no tens prou informació, digues-ho."},
            {"role": "user", "content": f"Context:\n{context_str}\n\nPregunta: {pregunta}"}
        ]
    )

    await update.message.reply_text(resposta.choices[0].message.content)

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    app = ApplicationBuilder().token(os.environ["TELEGRAM_TOKEN"]).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    app.run_polling()

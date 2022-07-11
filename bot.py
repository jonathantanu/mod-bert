import discord
from discord.ext import commands
from discord_slash import SlashCommand, SlashContext
from discord_slash.utils.manage_commands import create_option

from transformers import AutoTokenizer, BertForSequenceClassification
import torch
import numpy as np

# Config
client = commands.Bot(command_prefix="/")
slash = SlashCommand(client, sync_commands=True)
token = 'OTc2ODYwOTIxMTUyNTM2NjY2.GyTOdr.PeTN95Nj8mBz6ZCX7HrodwjGgC2CE4uFfM3MKE'
BERT_pretrained_model = "MyBert"

# Load Fine-Tuned model
print('Loading models...')
tokenizer = AutoTokenizer.from_pretrained(BERT_pretrained_model)
model = BertForSequenceClassification.from_pretrained(BERT_pretrained_model)
print('Models loaded!')

@client.event
async def on_ready():
    print('We have logged in as {0.user}'.format(client))

@client.event
async def on_message(message):
    # Make send to certain channe only
    # if message.channel.name == 'myberties':
    if message.author == client.user:
        return
    
    print(message.content)
    
    hate_speech = await _predictorFunction(message.content)

    if hate_speech != '':
        # Send message to channel with mentions
        await message.channel.send("WARNING! " + message.author.mention + " your message, " + "\"" + message.content + "\"" + " is classified as " + hate_speech + "!")
        
        # Send normal message
        # await message.channel.send(f"WARNING! " + "<@{user_id}>" + ", your message, " + "\"" + message.content + "\"" + " is " + hate_speech)
        hate_speech == ''

@slash.slash(
    name="Check",
    description="Check whether your statement contains hate speech or not",
    options=[
        create_option(
            name="statement",
            description="Write the statement you want to check",
            required=True,
            option_type=3
        )
    ]
)

# Send prediction text via slash command
async def _sendPrediction(ctx:SlashContext, statement:str):
    # message = await ctx.send(f"Checking your statement...")

    hate_speech = await _predictorFunction(statement)
     
    if hate_speech != '':
        print(hate_speech)
        await ctx.send("Your message, " + "\"" + statement + "\"" + " is " + hate_speech)
    else:
        await ctx.send("\""+ statement + "\"" + " is not an abusive sentence")

@slash.slash(
    name="help",
    description="Show help on how to use this bot"
)

async def _sendPrediction(ctx:SlashContext):
    await ctx.send("""
    Hello! I am a bot that can detect whether your statement contains hate speech or not.
    You can use /check command to check your statement. The bot will also check your message in the channel,
    and you will be warn if you use abusive words.
    """)


# BERT sentence prediction
async def _predictorFunction(sentence):
    inputs = tokenizer(sentence, return_tensors="pt")
    inputs = {k: v.to(model.device) for k,v in inputs.items()}

    logits = model(**inputs).logits

    sigmoid = torch.nn.Sigmoid()
    probs = sigmoid(logits.squeeze().cpu())
    predictions = np.zeros(probs.shape)
    predictions[np.where(probs >= 0.5)] = 1

    # Print labels of the prediction by the model
    predicted_labels = [model.config.id2label[idx] for idx, label in enumerate(predictions) if label == 1.0]
    predicted_string_list = ', '.join(map(str, predicted_labels))
    return predicted_string_list

client.run(token)
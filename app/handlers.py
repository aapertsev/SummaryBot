from aiogram import Router
from aiogram.filters import CommandStart
from app.summary import get_summary
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast



router = Router()


@router.message(CommandStart())
async def cmd_start(message):
    await message.reply("Hello, I'm a bot!")


@router.message(lambda message: message.text and len(message.text) > 300)
async def long_message(message):
    summary = await get_summary(summary_model, summary_tokenizer, message.text)
    await message.reply(summary)


# summary_model = BartForConditionalGeneration.from_pretrained("sn4kebyt3/ru-bart-large")
# summary_tokenizer = BartTokenizer.from_pretrained("sn4kebyt3/ru-bart-large")
model_name = "sn4kebyt3/ru-bart-large"
tokenizer = MBart50TokenizerFast.from_pretrained(model_name)
model = MBartForConditionalGeneration.from_pretrained(model_name)

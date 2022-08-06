#PL
#And Jos Because This is quite hard lol
#auto evil mean, annoying chatbot
#example -> 有人知道怎么修理bug吗 -> 怎么不去把你的人生修理好
#example -> 哈咯 -> 哈咯,废物
#example -> 我失恋了 -> 很好,继续寻找你的下一个失败吧

from neuralintents import GenericAssistant
from random import randint
import time

chatbot = GenericAssistant('chatbot.json')
chatbot.train_model()
chatbot.save_model()

async def input_chat(message,client):
    pass

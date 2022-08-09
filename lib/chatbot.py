#PL
#And Jos Because This is quite hard lol
#auto evil mean, annoying chatbot
#example -> 有人知道怎么修理bug吗 -> 怎么不去把你的人生修理好
#example -> 哈咯 -> 哈咯,废物
#example -> 我失恋了 -> 很好,继续寻找你的下一个失败吧

from random import randint
import time
import os
import sys
import discord

directory = os.getcwd()
sys.path.insert(0, directory+"\lib")
sys.path.insert(0, directory)
from neuralintents import GenericAssistant
chatbot = GenericAssistant('data/chatbot.json', model_name="data/DxB_Chatbot")
if not os.path.exists("data/DxB_Chatbot.h5"):
    import nltk
    nltk.download('omw-1.4')
    nltk.download("maxent_treebank_pos_tagger")
    nltk.download("maxent_ne_chunker")
    nltk.download("punkt")
    chatbot.train_model()
    chatbot.save_model()
else:
    chatbot.load_model()


async def input_chat(message,client):
    if message.author == client.user:
        return
    content = message.content

    # 1.basic example
    if content == '哈咯' or content == '嗨' or content == '你好':
        await message.channel.send(f"{content}，{message.author.mention}廢物 ☜(ﾟヮﾟ☜)")
        await message.add_reaction('👋')
    
    # 2. Special request or question 
    elif "你有朋友嗎" in content or "你有朋友吗？" in content:    														                                                                                       # 2.1 Friends
        await message.channel.send("跟你一樣沒朋友。")
        await message.add_reaction('🤣')
    elif "現在是什麽時間" in content or "現在時間是什麽" in content or "幾點了" in content or "现在是什么时间" in content or "现在时间是什么" in content or "几点了" in content or "what time" in content:        # 2.2 Current time
        now = time.localtime()
        current_time = time.strftime("%H:%M:%S", now)
        await message.channel.send(f"自己不會看時間哦😒😒\n算了現在時間是：{current_time}")
    elif "哈哈" in content:																                                                                                                                   # 2.3 React to laugh
        talk_2 = ["笑屁笑", "笑屁笑沒禮貌", "不好笑", "HAHAHAHAHAHAHA"]
        await message.add_reaction('😑')
        await message.channel.send(talk_2[randint(0, 3)])
    elif "錯" in content or "不對" in content or "错" in content or "不对" in content:									                                                                                    # 2.4 When Questioning bot
        if "不错" in content or "不錯" in content:
            await message.channel.send("哼")
        else:
            await message.channel.send("你才錯")
            await message.add_reaction('👎')
    elif "隨機數字" in content or "随机数字" in content or "random number" in content:									                                                                                       # 2.5 Random number
        if "from" in content or "從" in content or "到" in content or "从" in content:
            await message.channel.send(content="好麻煩還要定範圍...", components=[
                Button(style=ButtonStyle.URL,
                       label="自己來",
                       url="https://www.google.com/search?q=random+number")])
        else:
            await message.channel.send(randint(0, 100))
    elif "來個笑話" in content or "有笑話嗎" in content or "來个笑话" in content or "有笑话吗" in content:							                                                                             # 2.6 Make a joke
        await message.channel.send(f"{message.author.mention}👈👈🏻👈🏼👈🏽👈🏾👈🏿")
        await message.add_reaction('🤣')
    elif "dice" in content or "骰子" in content:														                                                                                                         # 2.7 Roll a dice
        dice = ["1", "2", "3", "4", "5", "6"]
        await message.channel.send(f"轉到了{dice[randint(0, 5)]}！")
    elif "你知道" in content or "是什麽" in content or "是什么" in content:										                                                                                                # 2.8 Just Rick-roll them
        embedVar = discord.Embed(title="不知道啦...", description=f"自己找[https://www.google.com?s={content}](https://www.youtube.com/watch?v=dQw4w9WgXcQ)")
        await message.channel.send(embed=embedVar)

    # 3.Encounter general question with Neuralintents
    else:
        response = chatbot.request(content)
        await message.channel.send(response)

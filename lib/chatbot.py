#PL
#And Jos Because This is quite hard lol
#auto evil mean, annoying chatbot
#example -> 有人知道怎么修理bug吗 -> 怎么不去把你的人生修理好
#example -> 哈咯 -> 哈咯,废物
#example -> 我失恋了 -> 很好,继续寻找你的下一个失败吧

first_run = False

from random import randint
import time
import os
import sys
import discord
from opencc import OpenCC

directory = os.getcwd()
sys.path.insert(0, directory+"\lib")
sys.path.insert(0, directory)
from neuralintents import GenericAssistant
chatbot = GenericAssistant('data/chatbot.json', model_name="data/DxB_Chatbot")
if not os.path.exists("data/DxB_Chatbot.h5") or first_run:
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
    cc = OpenCC('s2tw')
    content = cc.convert(message.content)

    # 1.basic example
    if content == '哈咯' or content == '嗨' or content == '你好':
        await message.channel.send(f"{content}，{message.author.mention}廢物 ☜(ﾟヮﾟ☜)")
        await message.add_reaction('👋')

    # 2. Special request or question
    # 2.1 Friends
    elif "你有朋友嗎" in content or "你有朋友吗？" in content:
        await message.channel.send("跟你一樣沒朋友。")
        await message.add_reaction('🤣')
    # 2.2 Current time
    elif "現在是什麽時間" in content or "現在時間是什麽" in content or "幾點了" in content or "现在是什么时间" in content or "现在时间是什么" in content or "几点了" in content or "what time" in content:
        now = time.localtime()
        current_time = time.strftime("%H:%M:%S", now)
        await message.channel.send(f"自己不會看時間哦😒😒\n算了現在時間是：{current_time}")
    # 2.3 React to laugh
    elif "哈哈" in content:
        talk_2 = ["笑屁笑", "笑屁笑沒禮貌", "不好笑", "HAHAHAHAHAHAHA"]
        await message.add_reaction('😑')
        await message.channel.send(talk_2[randint(0, 3)])
    # 2.4 choice
    elif " or " in content or "還是" in content or "还是" in content:
        choose = ["隨便啦！", "小孩子才做選擇。"]
        await message.channel.send(choose[randint(0, 1)])
    # 2.5 cannot
    elif "你可以" in content:
        cannot = ["不可以，麻煩。", "叫我做就做，將我不是很沒有面子？"]
        await message.channel.send(cannot[randint(0, 2)])
    # 2.6 When Questioning bot
    elif "錯" in content or "不對" in content or "错" in content or "不对" in content:
        if "不错" in content or "不錯" in content:
            await message.channel.send("哼")
        else:
            await message.channel.send("你才錯")
            await message.add_reaction('👎')
    # 2.7 Valorant Chinese Name
    elif "菲尼克斯" in content or "婕提" in content or "薇蝮" in content or "蘇法" in content or "瑟符" in content or "布史東" in content or "聖祈" in content or "歐門" in content \
            or "叛奇" in content or "芮茲" in content or "蕾娜" in content or "愷宙" in content or "絲凱" in content or "夜戮" in content or "亞星卓" in content or "錢博爾" in content \
            or "妮虹" in content or "菲德" in content or "特戰英豪" in content or "瓦羅蘭"in content or "无畏契约" in content:
        valorant = ["Don’t get in my way!","Welcome to my world!","Watch this!","Get out of my way!","Let’s go!","Off your feet!","Here comes the party!","Fire in the hole!",
                  "They will cower!","The hunt begins!","Come on, let’s go!","Jokes over, you’re dead!","Where is everyone hiding?","I know EXACTLY where you are.","Watch them run!",
                  "Scatter!","I am the hunter!","Nowhere to run!","Open up the sky!","Prepare for hellfire!","Your duty is not over!","You will not kill my allies!","Initiated!",
                  "You should run!","Seek them out!","“I’ve got your trail!","I’ll handle this","Who’s next","World divided","You’re divided","No one walks away","You. Are. Powerless",
                  "They are so dead","You want to play? Let’s play","Here we go!","Oy! I’m pissed!","Face Your Fear!","Nightmare take them","還不去放daily highlights?"]
        await message.channel.send(valorant[randint(0, 38)])
        await message.add_reaction('🔫')
    # 2.8 Random number
    elif "隨機數字" in content or "随机数字" in content or "random number" in content:
        if "from" in content or "從" in content or "到" in content or "从" in content:
            await message.channel.send(content="好麻煩還要定範圍...", components=[
                Button(style=ButtonStyle.URL,
                       label="自己來",
                       url="https://www.google.com/search?q=random+number")])
        else:
            await message.channel.send(randint(0, 100))
    # 2.9 Make a joke
    elif "來個笑話" in content or "有笑話嗎" in content or "來个笑话" in content or "有笑话吗" in content:
        await message.channel.send(f"{message.author.mention}👈👈🏻👈🏼👈🏽👈🏾👈🏿")
        await message.add_reaction('🤣')
    # 2.10 Roll a dice
    elif "dice" in content or "骰子" in content:
        dice = ["1", "2", "3", "4", "5", "6"]
        await message.channel.send(f"轉到了{dice[randint(0, 5)]}！")
    # 2.11 Just Rick-roll them
    elif "你知道" in content or "是什麽" in content or "是什么" in content:
        embedVar = discord.Embed(title="不知道啦...",
                                 description=f"自己找[https://www.google.com?s={content}](https://www.youtube.com/watch?v=dQw4w9WgXcQ)")
        await message.channel.send(embed=embedVar)
    # 2.12 Valorant -1
    elif "valorant -1" in content:
        await message.channel.send(f"@everyone")

    # 3.Encounter general question with Neuralintents
    else:
        response = chatbot.request(content)
        await message.channel.send(response)

        try_second_response = randint(0, 4)
        if try_second_response == 0:
            response2 = chatbot.request(content)
            if response2 != response:
                await message.channel.send(response2)

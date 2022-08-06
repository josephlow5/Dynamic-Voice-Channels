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
    elif content == '有人知道怎么修理bug吗':
        await message.channel.send("怎么不去把你的人生修理好")
    elif content == '我失恋了':
        await message.channel.send("很好,继续寻找你的下一个失败吧")

    # 2.Bot's general information in chinese
    elif "你的名字" in content or "你是什麽" in content or "你是什么" in content:										# 2.1 Name
        name = ["藍色的龍", "DragonXtBlue", "龍x藍", "秘密"]
        await message.channel.send(name[randint(0, 2)])
    elif "你的創作者" in content or "你的创作者" in content or "我在此鄭重發誓，我絕對不懷好意。" in content or "我在此郑重发誓，我绝对不怀好意。" in content:		# 2.2 Creator
        await message.channel.send("Josephlow、Shuhong、Jkkkk、Yohane和多多動物細胞\n自豪地獻上\n龍x藍")
    elif "你的功能" in content:																# 2.3 Skills
        function = ["目前的興趣是屌爆你。", "創造專屬你的語音頻道", "輸入'/'就知道了", "警告那些隨便mute和deafen的人",
                    "提醒大家來看live啊！", "秘密"]
        await message.channel.send(function[randint(0, 5)])
    elif "你幾歲" in content or "你几岁" in content: 													# 2.4 Age
        await message.channel.send("和你IQ一樣低")
    elif "你有朋友" in content:    															# 2.5 Friends
        await message.channel.send("跟你一樣沒朋友。")
        await message.add_reaction('🤣')
    elif "最喜歡" in content or "最愛" in content or "最喜欢" in content or "最爱" in content:    								# 2.6 Favourite and Hate
        if "不" in content:
            await message.channel.send("我不喜歡和你講話")
        else:
            await message.channel.send("我最喜歡嗆你")

    # 3.Other question or request in chinese
    						
    elif "问你" in content or "問你" in content:														# 3.1 Reject this kind of question
        await message.channel.send("不想回答ψ(｀∇´)ψ")

    # 3.2 Asking current time
    elif "現在是什麽時間" in content or "現在時間是什麽" in content or "幾點了" in content or "现在是什么时间" in content or "现在时间是什么" in content or "几点了" in content or "what time" in content:
        now = time.localtime()
        current_time = time.strftime("%H:%M:%S", now)
        await message.channel.send(f"自己不會看時間哦😒😒\n算了現在時間是：{current_time}")
    
    elif "吃什麽" in content or "選一個食物" in content or "吃什么" in content or "选一個食物" in content:							# 3.3 What are we eating today?
        food = ["屎", "shit", "一大坨屎", "椰漿飯", "雞飯"]
        await message.channel.send(f"{message.author}{food[randint(0, 4)]}好吃！")
    
    elif " or " in content or "還是" in content or "还是" in content:											# 3.4 Multiple Choice
        choice = ["隨便啦！", "小孩子才做選擇。"]
        await message.channel.send(choice[randint(0, 1)])

    # 3.5 Trash talking
    elif "幹" in content or "干" in content or "媽的" in content or "妈的" in content or "嫩" in content or "操你" in content or "雞掰" in content or "擊敗" in content or "鸡掰" in content or "鸡掰" in content or "不要吵" in content or "閉嘴" in content or "闭嘴" in content or "蛤" in content or "廢" in content or "废" in content or "屌" in content or "討厭" in content or "讨厌" in content:
        talk_1 = ["幹", "幹你媽", "操你媽", "雞掰", "？？？", "嫩！", "stfu", "shut the fk up", "shut up pls", "Fk you too",
                  "pussy", "shut up noob", "Jibai", "moron", "putang ina mo", "bobo"]
        await message.channel.send(talk_1[randint(0, 15)])
    
    elif "哈哈" in content:																# 3.6 React to laugh
        talk_2 = ["笑屁笑", "笑屁笑沒禮貌", "不好笑", "HAHAHAHAHAHAHA"]
        await message.add_reaction('😑')
        await message.channel.send(talk_2[randint(0, 3)])
    
    elif "錯" in content or "不對" in content or "错" in content or "不对" in content:									# 3.7 When Questioning bot
        if "不错" in content or "不錯" in content:
            await message.channel.send("哼")
        else:
            await message.channel.send("你才錯")
            await message.add_reaction('👎')
    
    elif "隨機數字" in content or "随机数字" in content or "random number" in content:									# 3.8 Random number
        if "from" in content or "從" in content or "到" in content or "从" in content:
            await message.channel.send(content="好麻煩還要定範圍...", components=[
                Button(style=ButtonStyle.URL,
                       label="自己來",
                       url="https://www.google.com/search?q=random+number")])
        else:
            await message.channel.send(randint(0, 100))
    
    elif "來個笑話" in content or "有笑話嗎" in content or "來个笑话" in content or "有笑话吗" in content:							# 3.9 Make a joke
        await message.channel.send(f"{message.author.mention}👈👈🏻👈🏼👈🏽👈🏾👈🏿")
        await message.add_reaction('🤣')
    
    elif "dice" in content or "骰子" in content:														# 3.10 Roll a dice
        dice = ["1", "2", "3", "4", "5", "6"]
        await message.channel.send(f"轉到了{dice[randint(0, 5)]}！")
    
    elif "你可以" in content:																# 3.11 Can you ...?
        await message.channel.send("不可以，麻煩。")
    
    elif "講什麽" in content or "講多一次" in content or "讲什么" in content or "讲多一次" in content:							# 3.12 When questioning bot's answer
        await message.channel.send("不要，麻煩。")
    
    elif "你知道" in content or "是什麽" in content or "是什么" in content:										# 3.13 Just Rick-roll them
        embedVar = discord.Embed(title="不知道啦...", description=f"自己找[https://www.google.com?s={content}](https://www.youtube.com/watch?v=dQw4w9WgXcQ)")
        await message.channel.send(embed=embedVar)

    # 4.Encounter English question with Neuralintents
    else:
        response = chatbot.request(content[0:])
        await message.channel.send(response)

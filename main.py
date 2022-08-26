import json
import os, sys
import random
import asyncio

#0. Import the latest discordpy (Not the stable, it is the version 2.0x)
directory = os.getcwd()
sys.path.insert(0, directory+"\lib")
import discord
from discord import app_commands
from discord.ext import tasks

#0.5 Import all submodules
from lib import data
from lib import vc_dynamic
from lib import vc_moderation
from lib import chatbot
from lib import s_command

#1. Read and Load Configs.
with open("data/config.json", encoding="utf-8") as configFile:
    config = json.load(configFile)
bot_token = config["token"]
bot_invite = config["invite"]


#2. Initialize Client Object (Note: To implement slash command, we must use CommandTree.
class MyClient(discord.Client):
    def __init__(self, *, intents: discord.Intents):
        super().__init__(intents=intents)
        self.tree = app_commands.CommandTree(self)       
    async def setup_hook(self):
        for connected_guild in client.guilds:
            #print(connected_guild.id)
            MY_GUILD = discord.Object(id=connected_guild.id)
            self.tree.copy_global_to(guild=MY_GUILD)
            await self.tree.sync(guild=MY_GUILD)
intents = discord.Intents.all()
client = MyClient(intents=intents)


#3. Slash Commands. (Moved to s_command.py)
s_command.slash_command_init(client)


#4. All kind of events.
@client.event
async def on_ready():
    await client.setup_hook()
    change_activity.start()
    print('DragonXtBlue Superb Bot is Fking Ready to Kick some ass!')

recent_chatbot_records = {}
@client.event
async def on_message(message):
    if message.author.id == client.user.id:
        return
    if message.interaction:
        return
    
    if data.is_monitor_chat_channel(message.channel.id):
        if message.channel.id not in recent_chatbot_records:
            recent_chatbot_records[message.channel.id] = {"timestamp":None}
        guild_record = recent_chatbot_records[message.channel.id]
        if guild_record["timestamp"] is not None:
            last_response = (message.created_at - guild_record["timestamp"]).total_seconds()
        else:
            last_response = 1000
        guild_record["timestamp"] = message.created_at
        if last_response > random.randint(1, 2):
            try:
                async with message.channel.typing():
                    await chatbot.input_chat(message,client)            #4.1 Chatbot.
            except Exception as e:
                await chatbot.input_chat(message,client)            
                
@client.event
async def on_voice_state_update(member, before, after):                
    if after.channel is not None:
        if data.is_monitor_channel(after.channel.id):
            await vc_dynamic.process_changes(member, after)      #4.2 Dynamic Voice Channels - Join VC.
    if before.channel is not None:
        if data.is_monitor_subchannel(before.channel.id): 
            if len(before.channel.voice_states.keys()) == 0:
                await vc_dynamic.close_subchannel(before)         #4.2 Dynamic Voice Channels - Leave VC.

    await vc_moderation.process_vc_action(member, before,        #4.3 Mute, Deafen, Stream - Action Moderation.
                                        after, client)
    

#5. Tasks
@tasks.loop(seconds=120.0)
async def change_activity():                                    #5.1 Dynamic Activity
    with open("data/activity.txt",encoding='utf-8') as activitiesFile:
        activities = activitiesFile.readlines()
    new_activities = []
    for activity in activities:
        if activity.strip("\n") != "":
            if activity.strip("\n") not in new_activities:
                new_activities.append(activity.strip("\n"))
    activities = new_activities
    try:
        selected = random.choice(activities)   
        activity = discord.Game(name=selected)
        await client.change_presence(status=discord.Status.online, activity=activity)
    except:
        pass


#6. Run the bot.
client.run(bot_token)

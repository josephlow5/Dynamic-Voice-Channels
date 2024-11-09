import discord
from discord import app_commands
import os
from dotenv import load_dotenv
import json
from openai import OpenAI
import time
import logging
import traceback
import psutil
import platform
from datetime import datetime, timezone
from typing import Optional, Any

# Load environment variables
load_dotenv()

# Set up Discord intents
intents = discord.Intents.default()
intents.message_content = True

# Initialize OpenAI client
openai_client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

class CustomClient(discord.Client):
    def __init__(self):
        super().__init__(intents=intents)
        self.tree = app_commands.CommandTree(self)
        self.system_prompt = self._load_system_prompt()
        self.conversation_history = {}
        self.model_settings = self.load_model_settings()
        self.listening_channels = self.load_listening_channels()
        self.rate_limits = self.load_rate_limits()
        self.last_message_time = {}
        self.length_limits = self.load_length_limits()
        self.processing_lock = {}  # Track processing status per channel
        self.setup_logging()
        self.command_aliases = {
            "h": "help",
            "settings": "show_settings",
            "channels": "show_channels",
        }
        
    async def setup_hook(self):
        await self.tree.sync()

    def load_model_settings(self):
        try:
            with open('model_settings.json', 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            default_settings = {
                "temperature": 1.0,
                "max_tokens": 2048,
                "top_p": 1.0,
                "frequency_penalty": 0.0,
                "presence_penalty": 0.0
            }
            self.save_model_settings(default_settings)
            return default_settings
            
    def save_model_settings(self, settings):
        with open('model_settings.json', 'w') as f:
            json.dump(settings, f, indent=2)

    def load_listening_channels(self):
        try:
            with open('listening_channels.json', 'r') as f:
                return set(json.load(f))
        except FileNotFoundError:
            self.save_listening_channels(set())
            return set()
            
    def save_listening_channels(self, channels):
        with open('listening_channels.json', 'w') as f:
            json.dump(list(channels), f)

    def _load_json(self, filename: str, default_value: Any) -> Any:
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            self._save_json(filename, default_value)
            return default_value

    def _save_json(self, filename: str, data: Any) -> None:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)

    def _load_system_prompt(self) -> str:
        try:
            with open('system_prompt.json', 'r', encoding='utf-8') as f:
                return json.load(f)['prompt']
        except FileNotFoundError:
            try:
                with open('default_prompt.txt', 'r', encoding='utf-8') as f:
                    default_prompt = f.read().strip()
                self._save_json('system_prompt.json', {"prompt": default_prompt})
                return default_prompt
            except FileNotFoundError:
                return "You are a helpful assistant."

    def load_rate_limits(self):
        try:
            with open('rate_limits.json', 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            default_limits = {"default": 3.0}  # 3 seconds default cooldown
            self.save_rate_limits(default_limits)
            return default_limits
            
    def save_rate_limits(self, limits):
        with open('rate_limits.json', 'w') as f:
            json.dump(limits, f, indent=2)

    def load_length_limits(self):
        try:
            with open('length_limits.json', 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            default_limits = {
                "default": {
                    "min": 1,
                    "max": 2000
                }
            }
            self.save_length_limits(default_limits)
            return default_limits
            
    def save_length_limits(self, limits):
        with open('length_limits.json', 'w') as f:
            json.dump(limits, f, indent=2)

    def setup_logging(self):
        # Create logs directory if it doesn't exist
        os.makedirs('logs', exist_ok=True)
        
        # Set up main logger
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f'logs/bot_{datetime.now().strftime("%Y%m%d")}.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger('discord_bot')
        
        # Set up transaction logger
        transaction_handler = logging.FileHandler('logs/transactions.log')
        transaction_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
        self.transaction_logger = logging.getLogger('transactions')
        self.transaction_logger.addHandler(transaction_handler)
        self.transaction_logger.setLevel(logging.INFO)

    def log_transaction(self, channel_id, user_id, input_text, response, model_settings, tokens_used=None, error=None):
        transaction = {
            'timestamp': datetime.now().isoformat(),
            'channel_id': channel_id,
            'user_id': user_id,
            'input_text': input_text,
            'response': response,
            'model_settings': model_settings,
            'tokens_used': tokens_used,
            'error': error
        }
        self.transaction_logger.info(json.dumps(transaction))

# Initialize the client
client = CustomClient()

@client.tree.command(name="health", description="Check bot's health status and metrics")
async def health_check(interaction: discord.Interaction):
    if str(interaction.user.id) != os.getenv('DEVELOPER_ID'):
        await interaction.response.send_message("You don't have permission to use this command!", ephemeral=True)
        return

    # Get system metrics
    process = psutil.Process()
    memory_use = process.memory_info().rss / 1024 / 1024  # Convert to MB
    cpu_use = process.cpu_percent()
    uptime = datetime.now(timezone.utc) - datetime.fromtimestamp(process.create_time(), tz=timezone.utc)

    # Get bot metrics
    total_channels = len(client.listening_channels)
    active_locks = sum(1 for locked in client.processing_lock.values() if locked)

    # Format status message
    status = f"""```
Bot Health Report
----------------
Status: 🟢 Online
Uptime: {str(uptime).split('.')[0]}
Memory Usage: {memory_use:.2f} MB
CPU Usage: {cpu_use:.1f}%
Python Version: {platform.python_version()}
Discord.py Version: {discord.__version__}

Channels:
- Listening to: {total_channels} channels
- Currently Processing: {active_locks} channels

Rate Limits:
- Default: {client.rate_limits.get('default', 'N/A')}s
- Custom Channels: {len(client.rate_limits) - 1}

Length Limits:
- Default: {client.length_limits['default']['min']}-{client.length_limits['default']['max']} chars
- Custom Channels: {len(client.length_limits) - 1}
```"""
    await interaction.response.send_message(status, ephemeral=True)
    

@client.tree.command(name="update_setting", description="Update a model setting")
async def update_setting(interaction: discord.Interaction, 
                        setting: str, 
                        value: float):
    if str(interaction.user.id) != os.getenv('DEVELOPER_ID'):
        await interaction.response.send_message("You don't have permission to use this command!", ephemeral=True)
        return
    
    valid_settings = ["temperature", "max_tokens", "top_p", "frequency_penalty", "presence_penalty"]
    if setting not in valid_settings:
        await interaction.response.send_message(f"Invalid setting. Valid options are: {', '.join(valid_settings)}", ephemeral=True)
        return
        
    if setting == "max_tokens":
        value = int(value)
    
    client.model_settings[setting] = value
    client.save_model_settings(client.model_settings)
    await interaction.response.send_message(f"Updated {setting} to {value}", ephemeral=True)

@client.tree.command(name="show_settings", description="Show all current model settings")
async def show_settings(interaction: discord.Interaction):
    if str(interaction.user.id) != os.getenv('DEVELOPER_ID'):
        await interaction.response.send_message("You don't have permission to use this command!", ephemeral=True)
        return
        
    settings_str = "Current Model Settings:\n```"
    settings_str += json.dumps(client.model_settings, indent=2)
    settings_str += "\n```\nSystem Prompt:\n```\n"
    settings_str += client.system_prompt
    settings_str += "\n```"
    
    await interaction.response.send_message(settings_str, ephemeral=True)

@client.tree.command(name="listen", description="Mark this channel as a listening channel")
async def listen(interaction: discord.Interaction):
    if str(interaction.user.id) != os.getenv('DEVELOPER_ID'):
        await interaction.response.send_message("You don't have permission to use this command!", ephemeral=True)
        return
        
    channel_id = str(interaction.channel_id)
    client.listening_channels.add(channel_id)
    client.save_listening_channels(client.listening_channels)
    await interaction.response.send_message("This channel is now marked as a listening channel!", ephemeral=True)

@client.tree.command(name="unlisten", description="Unmark this channel as a listening channel")
async def unlisten(interaction: discord.Interaction):
    if str(interaction.user.id) != os.getenv('DEVELOPER_ID'):
        await interaction.response.send_message("You don't have permission to use this command!", ephemeral=True)
        return
        
    channel_id = str(interaction.channel_id)
    if channel_id in client.listening_channels:
        client.listening_channels.remove(channel_id)
        client.save_listening_channels(client.listening_channels)
        await interaction.response.send_message("This channel is no longer a listening channel!", ephemeral=True)
    else:
        await interaction.response.send_message("This channel was not a listening channel!", ephemeral=True)

@client.tree.command(name="show_channels", description="Show all listening channels")
async def show_channels(interaction: discord.Interaction):
    if str(interaction.user.id) != os.getenv('DEVELOPER_ID'):
        await interaction.response.send_message("You don't have permission to use this command!", ephemeral=True)
        return
        
    if not client.listening_channels:
        await interaction.response.send_message("No channels are currently being listened to.", ephemeral=True)
        return
        
    channels_info = "Listening to these channels:\n```"
    for channel_id in client.listening_channels:
        channel = client.get_channel(int(channel_id))
        if channel:
            channels_info += f"\n#{channel.name} ({channel_id})"
    channels_info += "\n```"
    
    await interaction.response.send_message(channels_info, ephemeral=True)

@client.tree.command(name="set_rate_limit", description="Set rate limit (in seconds) for this channel")
async def set_rate_limit(interaction: discord.Interaction, seconds: float):
    if str(interaction.user.id) != os.getenv('DEVELOPER_ID'):
        await interaction.response.send_message("You don't have permission to use this command!", ephemeral=True)
        return
        
    channel_id = str(interaction.channel_id)
    client.rate_limits[channel_id] = seconds
    client.save_rate_limits(client.rate_limits)
    await interaction.response.send_message(f"Rate limit for this channel set to {seconds} seconds", ephemeral=True)

@client.tree.command(name="set_length_limit", description="Set message length limits for this channel")
async def set_length_limit(interaction: discord.Interaction, min_length: int, max_length: int):
    if str(interaction.user.id) != os.getenv('DEVELOPER_ID'):
        await interaction.response.send_message("You don't have permission to use this command!", ephemeral=True)
        return
    
    if min_length < 1 or max_length > 2000 or min_length > max_length:
        await interaction.response.send_message("Invalid limits. Min must be ≥1, max ≤2000, and min must be less than max.", ephemeral=True)
        return
        
    channel_id = str(interaction.channel.id)
    client.length_limits[channel_id] = {"min": min_length, "max": max_length}
    client.save_length_limits(client.length_limits)
    await interaction.response.send_message(f"Length limits set: min={min_length}, max={max_length}", ephemeral=True)

@client.event
async def on_message(message):
    # Ignore bot messages and interactions
    if message.author.bot or message.interaction:
        return
        
    # Check if this is a listening channel
    if str(message.channel.id) in client.listening_channels:
        channel_id = str(message.channel.id)
        
        # Check if we're already processing a message in this channel
        if client.processing_lock.get(channel_id, False):
            return  # Silently ignore if already processing
            
        # Set processing lock
        client.processing_lock[channel_id] = True
        
        try:
            # Check rate limit
            current_time = time.time()
            cooldown = client.rate_limits.get(channel_id, client.rate_limits["default"])
            
            if channel_id in client.last_message_time:
                time_passed = current_time - client.last_message_time[channel_id]
                if time_passed < cooldown:
                    return  # Silently ignore messages that come too quickly
                    
            client.last_message_time[channel_id] = current_time
            
            # Length limit check
            limits = client.length_limits.get(channel_id, client.length_limits["default"])
            msg_length = len(message.content)
            
            if msg_length < limits["min"]:
                await message.channel.send(f"Message too short! Minimum length is {limits['min']} characters.")
                return
                
            if msg_length > limits["max"]:
                await message.channel.send(f"Message too long! Maximum length is {limits['max']} characters.")
                return
            
            async with message.channel.typing():
                # Initialize or get conversation history
                if channel_id not in client.conversation_history:
                    client.conversation_history[channel_id] = []
                
                # Prepare messages including history
                messages = [{"role": "system", "content": client.system_prompt}]
                messages.extend(client.conversation_history[channel_id])
                messages.append({"role": "user", "content": message.content})
                
                # Calculate tokens (approximate)
                total_tokens = sum(len(m["content"]) / 4 for m in messages)
                
                # Trim history if needed
                while total_tokens > client.model_settings["max_tokens"] * 0.75 and len(client.conversation_history[channel_id]) > 0:
                    client.conversation_history[channel_id].pop(0)
                    messages = [{"role": "system", "content": client.system_prompt}]
                    messages.extend(client.conversation_history[channel_id])
                    messages.append({"role": "user", "content": message.content})
                    total_tokens = sum(len(m["content"]) / 4 for m in messages)
                
                response = openai_client.chat.completions.create(
                    model="ft:gpt-4o-mini-2024-07-18:personal::ARVpDpCJ",
                    messages=messages,
                    temperature=client.model_settings["temperature"],
                    max_tokens=client.model_settings["max_tokens"],
                    top_p=client.model_settings["top_p"],
                    frequency_penalty=client.model_settings["frequency_penalty"],
                    presence_penalty=client.model_settings["presence_penalty"]
                )
                
                answer = response.choices[0].message.content
                
                # Update conversation history
                client.conversation_history[channel_id].append({"role": "user", "content": message.content})
                client.conversation_history[channel_id].append({"role": "assistant", "content": answer})
                
                # Log successful transaction
                client.log_transaction(
                    channel_id=channel_id,
                    user_id=str(message.author.id),
                    input_text=message.content,
                    response=answer,
                    model_settings=client.model_settings,
                    tokens_used={
                        'prompt_tokens': response.usage.prompt_tokens,
                        'completion_tokens': response.usage.completion_tokens,
                        'total_tokens': response.usage.total_tokens
                    }
                )
                
                await message.channel.send(answer)
                
        except Exception as e:
            error_msg = f"Error processing message: {str(e)}\n{traceback.format_exc()}"
            client.logger.error(error_msg)
            
            # Log failed transaction
            client.log_transaction(
                channel_id=channel_id,
                user_id=str(message.author.id),
                input_text=message.content,
                response=None,
                model_settings=client.model_settings,
                error=str(e)
            )
            
            await message.channel.send("Sorry, I encountered an error. The error has been logged for review.")
            
        finally:
            # Always release the lock when done, even if there was an error
            client.processing_lock[channel_id] = False

@client.event
async def on_ready():
    client.logger.info(f'Logged in as {client.user.name}')
    try:
        synced = await client.tree.sync()
        client.logger.info(f"Synced {len(synced)} command(s)")
    except Exception as e:
        client.logger.error(f"Error syncing commands: {e}")

@client.event
async def on_error(event, *args, **kwargs):
    client.logger.error(f"Error in {event}:")
    client.logger.error(traceback.format_exc())

@client.tree.command(name="help", description="Show available commands and their usage")
async def help_command(interaction: discord.Interaction, command: Optional[str] = None):
    if str(interaction.user.id) != os.getenv('DEVELOPER_ID'):
        await interaction.response.send_message("You don't have permission to use this command!", ephemeral=True)
        return

    commands = {
        "help": "Show this help message. Usage: /help [command]",
        "health": "Check bot's health status and metrics",
        "update_setting": "Update a model setting. Usage: /update_setting <setting> <value>",
        "show_settings": "Show all current model settings",
        "listen": "Mark current channel as a listening channel",
        "unlisten": "Unmark current channel as a listening channel",
        "show_channels": "Show all listening channels",
        "set_rate_limit": "Set rate limit for current channel. Usage: /set_rate_limit <seconds>",
        "set_length_limit": "Set message length limits. Usage: /set_length_limit <min_length> <max_length>"
    }

    if command:
        # Check aliases first
        if command in client.command_aliases:
            command = client.command_aliases[command]
            
        if command in commands:
            await interaction.response.send_message(f"```\n/{command}: {commands[command]}\n```", ephemeral=True)
        else:
            await interaction.response.send_message(f"Command '{command}' not found.", ephemeral=True)
    else:
        help_text = "Available Commands:\n```"
        for cmd, desc in commands.items():
            help_text += f"\n/{cmd}: {desc}"
        help_text += "\n\nAliases:"
        for alias, cmd in client.command_aliases.items():
            help_text += f"\n/{alias} → /{cmd}"
        help_text += "```"
        await interaction.response.send_message(help_text, ephemeral=True)

# Run the client
client.run(os.getenv('DISCORD_TOKEN'))
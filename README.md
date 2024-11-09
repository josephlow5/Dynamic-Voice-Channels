# DragonXtBlue-Impressive (NLP AI Sarcastic Responding Discord Bot)

Inspired by https://yeecord.com/

## Original Features

1. **Dynamic Voice Channels** - No more messy voice channels, create one when someone is needed.

2. **Voice Channel Activity Moderation** - Kick, warn when people mute or deafen. Send someone in to cheers while someone starts Go Live.

3. **Chatbot** - Feeling lonely no one is having conversation with you? Not to worry, we have some 'REAL HUMAN' staff to chat with you.

4. **Slash Command Setup** - Nothing to describe.

## New Features

1. **Intelligent Chat Responses**
   - Powered by OpenAI's GPT model
   - Maintains conversation history per channel
   - Configurable response parameters
   - Smart token management

2. **Channel Management**
   - Selective channel listening
   - Rate limiting per channel
   - Message length restrictions
   - Processing lock to prevent message overlap

3. **Administrative Controls**
   - Developer-only command access
   - Customizable model settings
   - Health monitoring system
   - Comprehensive logging system

4. **Slash Commands**
   - `/health` - View bot metrics and status
   - `/update_setting` - Modify model parameters
   - `/show_settings` - Display current configuration
   - `/listen` & `/unlisten` - Manage active channels
   - `/set_rate_limit` - Control message frequency
   - `/set_length_limit` - Set message size boundaries
   - `/help` - Command documentation

## Installation

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   (requirements.txt is in data folder)

2. Install [vcredist](https://support.microsoft.com/help/2977003/the-latest-supported-visual-c-downloads)

3. Change variable **first_run** to ***True*** in lib/chatbot.py

4. Put your bot info in data/config.json

5. Run main.py once and change back **first_run** to ***False*** in lib/chatbot.py

6. Set up environment variables in `.env`:
   ```
   DISCORD_TOKEN=your_discord_token
   OPENAI_API_KEY=your_openai_key
   DEVELOPER_ID=your_discord_id
   ```

7. Create necessary JSON configuration files:
   - `model_settings.json`
   - `listening_channels.json`
   - `rate_limits.json`
   - `length_limits.json`
   - `system_prompt.json`

## Configuration

The bot uses several JSON files for configuration:

- `model_settings.json`: OpenAI model parameters
- `listening_channels.json`: Active channel IDs
- `rate_limits.json`: Message cooldown settings
- `length_limits.json`: Message size restrictions
- `system_prompt.json`: Bot's system instructions

## Logging

The bot maintains two types of logs:
- Daily bot logs in `logs/bot_YYYYMMDD.log`
- Transaction logs in `logs/transactions.log`

## Requirements

- Python 3.8+
- discord.py
- openai
- python-dotenv
- psutil

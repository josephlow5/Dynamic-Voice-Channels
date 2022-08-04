#Shuhong
#Add Category
#Create Monitor Channel -> add to data
#Reply slash command

@client.command()
@has_permissions(kick_members=True)
async def warn(ctx, member:discord.Member, *, reason=None):
 arg=reason
 author=ctx.author
 guild=ctx.message.guild
 overwritee = discord.PermissionOverwrite()
 overwrite = discord.PermissionOverwrite()
 channel = get(guild.text_channels, name='warn-logs')
 category = get(guild.category_channels, name='Multi-Logs')
 mrole = get(ctx.guild.roles, name="Multi-Galaxy")

 if category is None:
  category = await guild.create_category_channel("Multi-Logs")
  overwritee.read_messages = False
  overwritee.read_message_history = False
  overwritee.send_messages = False
  overwrite.read_messages = True
  overwrite.read_message_history = True
  overwrite.send_messages = True
  await channel.set_permissions(guild.default_role, overwrite=overwritee)
  await channel.set_permissions(mrole, overwrite=overwrite)
 if channel is None:
  channel = await guild.create_text_channel('warn-logs')
  overwritee.read_messages = False
  overwritee.read_message_history = False
  overwritee.send_messages = False
  overwrite.read_messages = True
  overwrite.read_message_history = True
  overwrite.send_messages = True
  await channel.set_permissions(guild.default_role, overwrite=overwritee)
  await channel.set_permissions(mrole, overwrite=overwrite)
async def setup(guild,client):
    reply = "Dynamic Voice Channels setup successfully!"
    return reply

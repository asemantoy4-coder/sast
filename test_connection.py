import utils
import config

print("🔗 Testing connection to Telegram...")
success = utils.send_telegram_notification("⚡️ *Aseman Bot Connected Successfully!*\nReady to scan the market.")

if success:
    print("✅ Success! Check your Telegram channel.")
else:
    print("❌ Failed! Please check your Token, Chat ID, and make sure the bot is an ADMIN in your channel.")
import telegram
import argparse


def send_update_via_telegram(message):
    bot = telegram.Bot(token="")
    bot.sendMessage(chat_id=None, text=message)


def parse_options():
    parser = argparse.ArgumentParser()
    parser.add_argument('--message', type=str, default='', help='message to send to bot')
    opt = parser.parse_args()
    return opt


if __name__ == '__main__':
    options = parse_options()
    send_update_via_telegram(message=options.message)

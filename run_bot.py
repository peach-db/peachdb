import shelve

from peachdb.bots.qa import QABot
from peachdb.constants import CONVERSATIONS_DB

bot = QABot(
    bot_id="my_bot_12",
    embedding_model="openai_ada",
    llm_model_name="gpt-3.5-turbo",
    system_prompt="Please answer questions about the 1880 Greenback Party National Convention.",
)

bot.add_data(
    documents=[
        "The 1880 Greenback Party National Convention convened in Chicago from June 9 to June 11 to select presidential and vice presidential nominees and write a party platform for the Greenback Party in the United States presidential election of 1880. Delegates chose James B. Weaver of Iowa for President and Barzillai J. Chambers of Texas for Vice President.",
        'The Greenback Party was a newcomer to the political scene in 1880 having arisen, mostly in the nation\'s West and South, as a response to the economic depression that followed the Panic of 1873. During the Civil War, Congress had authorized "greenbacks", a form of money redeemable in government bonds, rather than in then-traditional gold. After the war, many Democrats and Republicans in the East sought to return to the gold standard, and the government began to withdraw greenbacks from circulation. The reduction of the money supply, combined with the economic depression, made life harder for debtors, farmers, and industrial laborers; the Greenback Party hoped to draw support from these groups.',
    ]
)

cid, answer = bot.create_conversation_with_query("Where did the convention convene?")

answer_2 = bot.continue_conversation_with_query(cid, "Who was the vice presidential nominee?")

answer_3 = bot.continue_conversation_with_query(cid, "What was The Greenback Party?")

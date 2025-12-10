import pandas as pd
import re
from tqdm import tqdm
from Impact.Models.Llama3_1_Instruct.Llama3_1_Instruct import LlamaInstruct

# 1. Initialize the Model
# We use your existing factory pattern
LlamaInstructFactory = LlamaInstruct()
llm = LlamaInstructFactory.create()

# 2. Define the Optimized System Prompt
# We remove all instructions about JSON and Reasoning to save tokens.
system_instruction = """
You are an expert financial analyst specializing in the US Economy. 
Analyze the news headline and determine its relevance to the 'US Economy'.

Scoring Rules (0.0 to 1.0):
- 0.9-1.0: Direct US Macro impact (Fed rates, US Inflation, US GDP, Major US regulations).
- 0.7-0.8: Significant US Corporate News (Mergers of US firms, Antitrust involving US firms like Nvidia/Google, US Labor strikes).
- 0.3-0.5: Global news with indirect US impact (Oil prices, Global supply chains).
- 0.0-0.2: Irrelevant (Foreign domestic news, UK/EU specific with no US spillover, Sports, Quizzes, localized crimes).

Crucial Relationships to Spot:
- 'Fed' / 'Federal Reserve' -> High Relevance.
- '£' or 'Euro' symbols often imply UK/EU context -> Low Relevance (unless global).
- 'Nvidia', 'Apple', 'Tesla' are US companies -> Moderate/High Relevance.
- 'Protests in Peru', 'UK Inflation' -> Low Relevance.

Output ONLY the float number. Do not explain.
"""


def get_relevance_score(headline):
    """
    Fast inference returning only a float.
    """
    prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
{system_instruction}<|eot_id|><|start_header_id|>user<|end_header_id|>
Headline: "{headline}"<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""

    output = llm(
        prompt,
        max_tokens=6,  # Hard limit: Generate only the number (e.g., "0.7")
        stop=["<|eot_id|>", "\n"],  # Stop immediately after the number
        temperature=0.0,  # Deterministic (0.0 is faster/safer for classification)
        echo=False,
    )

    try:
        # Extract text
        raw_text = output['choices'][0]['text'].strip()

        # Fast regex to find a float (e.g., "0.7", "1.0", "0")
        match = re.search(r"(\d+(\.\d+)?)", raw_text)

        if match:
            return float(match.group(1))
        return 0.0
    except Exception:
        return 0.0


# --- Your Data ---
headlines = [
    "Energy watchdog to cap energy bills by limiting profit of networks",
    "Every McDonald's in Peru closes amid protests at death of two workers",
    "Advertising watchdog bans e-cigarette promotion on Instagram",
    "Zero-carbon ships on horizon under fuel levy plan",
    "Nils Pratley on finance That Persimmon £75m bonus now looks as shoddy as its homes",
    "Dish Network Corp Chairman Charlie Ergen testified on Wednesday that the Justice Department's antitrust chief advised him on June 10 to ask a senator to speak to the Federal Communications Commission about approving a key piece of the merger of wireless carriers T-Mobile and Sprint.",
    "Swatch Group threatened on Wednesday to pursue damages after it said the Swiss competition authority COMCO wants to prohibit the world's biggest watch maker from supplying watch mechanisms to other companies during 2020.",
    "U.S. chipmaker Nvidia is set to win unconditional EU antitrust approval for its $6.8 billion acquisition of Mellanox Technologies , people familiar with the matter said on Wednesday.",
    "Daimler AG's Mercedes-Benz USA has agreed to a $20 million civil penalty over its handling of U.S. vehicle recalls after a year-long U.S. government investigation into 1.4 million recalled vehicles.",
    "Fiat Chrysler Chief Executive Mike Manley will remain with the new group set to result from a planned merger with French rival PSA-Peugeot, Chairman John Elkann said on Wednesday.",
    "Fiat Chrysler has its own goals to meet next year, independently of its planned merger with Peugeot owner PSA, Chief Executive Mike Manley said on Wednesday.",
    "US teens may be barred from buying vape pens and cigarettes",
    "Guardian business Christmas quiz 2019",
    "UK inflation stays low despite rising cost of chocolate and holidays",
    "Project Syndicate economists Six tax-based ways to tackle US income inequality",
    "Retailer Beales puts itself on the market in 'challenging times'",
    "Fiat Chrysler and Peugeot maker PSA on Wednesday reached a binding agreement for their roughly $50 billion merger, as the auto industry scrambles to develop zero-emissions vehicles and tackle slowing demand.",
    "PM bans ministers from Davos in nod to working-class voters",
    "Persimmon accused of building homes with 'intolerable' fire risk",
    "Vera Lynn stops name being used to market gin in trademark ruling",
    "Pound slides to pre-election levels in wake of bid to outlaw Brexit extension",
    "Book People goes into administration, with almost 400 jobs at risk",
    "UK manufacturing output falling at fastest rate since 2009, says CBI",
    "UK estate agents fined £600,000 for price-fixing",
    "Whirlpool recalls more than 500,000 washing machines over fire risk",
    "Business live Pound tumbles amid Brexit cliff-edge fears; factory output slumps - as it happened",
    "NAB chief admits banks' drive for profit helped create industry crisis",
    "Fiat Chrysler and Peugeot maker PSA are pushing ahead with a planned merger worth about $50 billion aimed at creating the world's No. 4 carmaker.",
    "Insurer MetLife Inc. will pay $10 million to settle with U.S. regulators over longstanding internal controls failures, the Securities and Exchange Commission (SEC) said on Wednesday.",
    "Britain's biggest union is seeking guarantees on the long-term future of Peugeot's plants in Britain after the French carmaker struck a binding deal on Wednesday with Fiat Chrysler to create the world's fourth-biggest carmaker.",
    "U.S. retailers are expected to ring up record sales on Super Saturday this year, as fewer days than usual between Thanksgiving and Christmas have squeezed shoppers to finish their purchases.",
    "Boeing Co's decision to suspend aerospace's biggest production line exposes contrasts in the U.S.-dominated 737 MAX supply chain, severely straining some niche machine shops while giving engine giants time to iron out their own wrinkles.",
    "British American Tobacco (BAT) must stop advertising its e-cigarettes from any public account on Instagram, including some influencers' accounts, the UK's advertising watchdog ruled on Wednesday.",
    "U.S. Federal Reserve Governor Lael Brainard launched a broadside against Facebook's Libra digital currency project on Wednesday, saying it faces a core set of legal and regulatory challenges including clarity about how it would be tied to some basket of underlying assets.",
    "Chancellor Angela Merkel's conservatives and their Social Democrat partners have delayed until next year a decision on security rules for Germany's 5G network that could bar China's Huawei, a highly divisive issue in an unhappy alliance.",
    "German Chancellor Angela Merkel said on Wednesday she had not been told about the Chinese authorities making any threats of retaliation if Germany were to exclude Huawei from its 5G rollout.",
    "Saudi Aramco shares fell almost 3% on Wednesday as investors pocketed profits on the day it was included in the MSCI Emerging Markets Index and the Saudi Tadawul index .",
    "Uber driver Juan Jose is quick to cover his cell phone with the red rag he keeps in his front seat whenever he spots a police officer on the streets of Bogota.",
    "Amazon.com Inc , Apple Inc and Alphabet Inc's Google are partnering to lay the groundwork for better compatibility among their smart home products, the companies said on Wednesday.",
    "American International Group Inc on Thursday named Peter Zaffino, the executive overseeing a turnaround effort of the company's general insurance unit, as the company's president.",
    "Fiat Chrysler Automobiles (FCA) and Peugeot owner PSA have sealed a deal to join forces in a 50-50 share merger worth about $50 billion.",
    "(This Dec. 17 story corrects translation in para 13, replacing 'abuse' with 'generalisation' in reference to the concept of national security)",
    "Italy's Finance Minister Roberto Gualtieri on Wednesday welcomed the binding merger agreement between Fiat Chrysler and Peugeot maker PSA saying it was good and said Rome would keep on eye on the deal's effects on jobs and investments.",
    "Fiat Chrysler (FCA) and Peugeot maker PSA merger is good news for France, Europe and also for the car industry, France's Finance Minister Bruno Le Maire told Reuters on Wednesday.",
    "Fiat Chrysler will meet Italian trade unions in Turin on Friday to discuss the details of a merger deal with French rival PSA , the FIM-CISL union said on Wednesday.",
    "Alphabet Inc's Google has settled a longstanding tax dispute with Australia's tax office, it said on Wednesday, after paying an extra A$481.5 million ($326.75 million) on top of its previous tax bill.",
    "U.S. electric vehicle maker Tesla Inc is considering cutting the prices of its China-built Model 3 sedans by 20% or more next year, Bloomberg reported on Wednesday, citing people familiar with the plans.",
    "Trade groups representing some of the world's biggest firms plan to lobby U.S. officials and Indian lawmakers in a bid to dilute parts of an Indian privacy bill which could hurt businesses, three sources familiar with the plans told Reuters.",
    "Lenovo Group's founder, Liu Chuanzhi, will retire at the year end as chairman of its parent Legend Holdings, the company said on Wednesday.",
    "Fruit smoothies may become the new martini for Wall Street holiday parties.",
    "JPMorgan said on Wednesday it has received the final approval from Chinese regulators to set up a majority-owned securities venture in the country.",
    "U.S. President Donald Trump's rewrite of North American trade rules will cost automakers nearly $3 billion more in tariffs over the next decade for cars and parts that will not meet higher regional content requirements over the next decade, the Congressional Budget Office (CBO) estimates.",
    "A U.S. bankruptcy judge approved on Tuesday PG&E Corp's $13.5 billion settlement with victims of Californian wildfires, and the company's stock rallied as the utility gained momentum to emerge from bankruptcy in June as planned.",
    "All 29 McDonald's Peru locations will remain closed until its local operator Arcos Dorados Holdings Inc completes inspections following the deaths of two teenaged employees at the weekend, the franchisor said in a statement on Wednesday.",
    "Airbus is on course to end 2019 with a rise in its order backlog after netting more sales than deliveries across its major products, a senior executive said on Wednesday, thanks partly to strong demand in Asia.",
    "The S&P 500 ended a five-day winning streak on Wednesday as investors' optimism about global economic growth was countered by a steep drop in FedEx Corp shares, but the benchmark index managed to hover near all-time highs.",
    "FedEx Corp shares tumbled more than 10% on Wednesday after the company slashed its 2020 profit forecast a second time, as it revamps its business to replace slumping air shipments with lower-profit residential deliveries.",
    "With a track record of streamlining Peugeot's portfolio of vehicles, engines and platforms and offering generous layoffs Carlos Tavares has a ready-made manual for combining France's most profitable carmaker with Fiat Chrysler (FCA).",
    "A division of SNC-Lavalin Group Inc pleaded guilty to one fraud charge and will pay a C$280 million fine related to projects in Libya, the company said on Wednesday, in a case that engulfed Canadian Prime Minister Justin Trudeau's government in crisis.",
    "Freddie Mac has offered early retirement to around 25% of its staff as it begins to overhaul its workforce amid a broader push by the Trump administration to reform the housing finance giant, according to four people briefed on the matter.",
    "Tariffs and trade tensions are a huge source of worry for U.S. companies, with nearly half of Fortune 500 companies referencing such concerns during last quarter's earnings calls, the U.S. Chamber of Commerce said on Thursday.",
    "The dollar gained on Wednesday as improving economic data squashed the likelihood of a Federal Reserve interest rate cut in 2020, while a rally in global equity markets wavered as financials shares slipped.",
    "Oil prices steadied on Wednesday after U.S. government data showed a decline in crude inventories and on expectations for an uptick in demand next year on the back of progress in resolving the U.S.-China trade fight.",
    "A dam owned by iron ore company Vale SA that was the subject of an investigative report by a TV program last week is structurally sound and there is no reason for concern, Brazil's national mining regulator said late on Tuesday.",
    "Unicorns like Uber, Lyft and Slack may have had disappointing IPOs, but U.S. venture capital firms gave birth to a record number of unicorns in 2019.",
    "Cyrus Mistry, who has been embroiled in a legal battle with Tata Group since being ousted in 2016, won backing from a tribunal to be reinstated as executive chairman of its holding company, lawyers said on Wednesday.",
    "U.S. homesharing site Airbnb will get a better idea of how it should be labeled when Europe's top court rules on Thursday on the tricky question of whether it is an online booking service or a real estate agent subject to onerous rules.",
    "Billed as a merger of equals, PSA's $50 billion tie-up with Fiat Chrysler (FCA) gives the Peugeot owner one potentially big advantage, its own boss will be firmly behind the new wheel.",
    "Microsoft is on a rocketship this year, but its stock charts indicate it could come back down to Earth soon.",
    "CNBC's Jim Cramer breaks down the positive signs in the latest housing data, sits down with Eli Lilly's CEO and reveals stocks worth buying.",
    "Mad Money host Jim Cramer rings the lightning round bell, which means he's giving his answers to callers' stock questions at rapid speed.",
    "The Mad Money host says there's no need to worry on Wall Street, even if the market has more down days ahead.",
    "In 1999, President Bill Clinton's impeachment was irrelevant [to stocks]. You had to use any weakness to buy and then to buy and then to buy, the Mad Money host says.",
    "It's really pretty astonishing how much of what was working then is also working now, the Mad Money host says.",
    "Stop the predictions, stop the sham guidance, and take the time to figure it all out. Then, and only then, will FedEx the stock be able to bottom, the Mad Money host says.",
    "CNBC's Jim Cramer explained why the impeachment process won't disturb investing on Wall Street. The Mad Money host also breaks down what's plaguing FedEx.",
    "Fossil fuels fall to record low proportion of UK energy mix",
    "Bank of England keeps interest rates on hold despite weak economy",
    "Conscious, ethical and cruelty-free: a guide to the language of sustainable fashion",
    "EU court rules Airbnb does not require estate agent licence",
    "‘It’s just weird’: Oshawa sends off GM plant as thousands scramble for jobs",
    "Retail sales at 19-month low as Christmas shoppers leave it late",
    "'It's mind-blowing': Emma Tillinger Koskoff on producing The Irishman and Joker",
    "British Airways slumps to near bottom in passenger survey",
    "Nils Pratley on finance Bet365 chief's pay is less fascinating than where its revenues are made",
    "Separate scandal-hit UK audit sector from accounting, says report",
    "Google and Facebook dominance should be curbed, suggests CMA",
    "UK banks and insurers to be tested on climate crisis response plans",
    "Business live UK inflation at three-year low; Brexit worries weigh - as it happened",
    "Bet365 boss Denise Coates hits jackpot with £323m payday",
    "Nils Pratley on finance Bank of England deserves to be embarrassed about security breach",
    "Debt in developing economies rises to record $55tn",
    "What is high-frequency trading and how do you make money from it?",
    "Rightwing news network Sinclair raises minimum wage to $15 an hour",
    "Unite union seeks jobs assurances after PSA-Fiat Chrysler merger",
    "UK house price growth to remain low despite talk of 'Boris bounce'",
    "Volkswagen's namesake core brand is on track to post a record operating profit this year thanks to cost savings and its increased sales of sports utility vehicles (SUV's), a senior manager of the unit said.",
    "Semiconductor company Mellanox Technologies has received approval from EU antitrust and Mexico for its proposed acquisition by chipmaker Nvidia for $6.8 billion, a regulatory filing showed on Thursday.",
    "Agreements that let Facebook and other firms send European citizens' data to the United States and other countries are valid, a key EU court adviser said on Thursday, although he left room for such transfers to be blocked if European data protection standards are not met in countries receiving the information.",
    "Apple Inc executives met James Bond franchise-owner MGM Holdings Inc and the collegiate athletic conference Pac-12 earlier this year as part of its efforts to boost the Apple TV service, the Wall Street Journal reported https://on.wsj.com/2PEGhH7 on Thursday.",
    "The dollar was stalled on Thursday a day ahead of the release of U.S. gross domestic product data, little moved by weak factory activity data or President Donald Trump's impeachment.",
    "U.S. companies' borrowings for capital investments fell about 3% in November from a year earlier, the Equipment Leasing and Finance Association (ELFA) said on Thursday.",
    "Oil prices reached the highest level in three months in thin pre-Christmas trading on Thursday, buoyed by the previous day's news that U.S. crude inventories declined and as U.S.-China trade tensions continued to ease.",
    "BMW AG and Daimler AG said Wednesday they plan to exit the North American car-sharing market, with the joint venture partners halting operations in Montreal, New York, Seattle, Washington, D.C., and Vancouver, as they focus on the European market.",
    "France's 'AHTOP' tourism association urged the French government to take action after U.S. homesharing site Airbnb won its battle to remain exempt from onerous European property regulations.",
    "U.S. homesharing site Airbnb on Thursday won its battle to remain exempt from onerous European property regulations, as the EU's top court ruled it was an online platform and not a property agent.",
    "Starbucks Corp agreed on Thursday to pay restitution and accept greater oversight to settle a multi-year probe finding that it had illegally required New York City employees to find substitutes when they needed to use sick leave.",
    "Goldman Sachs Group Inc is in talks with the U.S. government and a state regulator to possibly pay up to $2 billion and admit guilt to resolve investigations into its role in the 1MDB Malaysian corruption scandal, according to a source familiar with the matter.",
    "Global equity markets extended a year-end rally on Thursday that has pushed U.S. and world stock benchmarks to record highs, while bond yields in Europe rose after Sweden stopped five years of negative interest rates, signaling the end of a sub-zero era.",
    "A federal judge on Thursday rejected Southwest Airlines Co's bid to dismiss a discrimination lawsuit by an American of Iraqi descent who was removed from a 2016 flight after another passenger heard him speak in Arabic and feared he might be a terrorist.",
    "Amazon.com Inc said on Thursday it was on track to deliver 3.5 billion customer packages globally this year through its in-house delivery network.",
    "Brazil's planemaker Embraer S.A. said it is studying the development of a new light military transport plane together with the Brazilian Air Force, the company said in a statement on Thursday.",
    "A U.S. congressional committee will ask Wells Fargo & Co’s board of directors to testify about the bank's sales scandal next year, a senior Democrat told Reuters on Thursday.",
    "Boeing Co is set to launch its new astronaut capsule on Friday on its first unmanned journey to the International Space Station, a milestone test for the U.S. aerospace firm that is vying with SpaceX to revive NASA's human spaceflight capabilities.",
    "Nike Inc's quarterly revenue and profit blew past Wall Street expectations on Thursday on strong sales in China, but lower-than-expected growth in North America, its biggest market, overshadowed the beat.",
    "Australia's second-largest lender Westpac Banking Corp on Friday appointed two people to a panel which will assess the bank's accountability in a money laundering scandal.",
    "The general counsel of a U.S. labor agency has accused Chipotle Mexican Grill Inc of violating U.S. labor law by allegedly firing an employee in New York in retaliation for complaining about workplace problems and trying to organize with a union.",
    "The U.S. House of Representatives overwhelmingly approved a new North American trade deal on Thursday that includes tougher labor and automotive content rules but leaves $1.2 trillion in annual U.S.-Mexico-Canada trade flows largely unchanged.",
    "Australia's competition regulator said the federal court has penalized Volkswagen AG a record A$125 million ($86 million) for allegedly making false representations about compliance with the country's diesel emission standards.",
    "The new North American trade deal addresses most concerns of U.S. labor unions, Richard Trumka, head of the AFL-CIO union federation, said on Thursday, but even so he expects it could take more than a decade to reverse job losses resulting from the original 25-year-old trade pact.",
    "Mexican Foreign Minister Marcelo Ebrard said on Thursday that a new North American trade deal will start a new phase of investment and growth for Mexico, ending a period of uncertainty.",
    "The number of Americans filing applications for unemployment benefits dropped from a more than two-year high last week, pointing to sustained labor market strength that should continue to underpin consumer spending and the economy.",
    "The U.S. Securities and Exchange Commission on Wednesday proposed changes to its decades-old definition of a professional investor in order to allow more Americans to buy shares in private companies.",
    "A German court on Thursday banned Uber ride-hailing services in Germany, arguing the U.S. company lacks a necessary licence to offer passenger transport services using rental cars.",
    "China and the United States are in touch over the signing of their Phase 1 trade deal, China's commerce ministry said, which will see lower U.S. tariffs on Chinese goods and higher Chinese purchases of U.S. farm, energy and manufactured goods.",
    "Top Indian electricity generator NTPC has rejected the emissions-cutting technology of GE and other foreign firms for its coal-fired plants, documents show, shutting them out of an estimated $2 billion in orders.",
    "Philippines' Cebu Air Inc said on Thursday it had ordered 15 Airbus SE A320neo family jets worth $2 billion at list prices, keeping the European manufacturer on track to reach a year-end sales milestone.",
    "E-commerce giant Alibaba Group Holding Ltd and its fintech affiliate Ant Financial announced a string of top level executive changes on Thursday, amid an ongoing restructuring following the appointment of Daniel Zhang as chairman of Alibaba.",
    "Britain's Financial Reporting Council (FRC) has added an extra year to its investigation into accountant EY's audit of Thomas Cook, the holiday company that collapsed in September leaving tens of thousands of people stranded abroad.",
    "Incoming Chief Executive Bernard Looney will not take up either of BP's seats on the board of Rosneft when he takes over in February because of the complexity of the relationship with the Russian oil giant, five company sources said.",
    "SoftBank-controlled Yahoo Japan has pulled out of a Japanese apartment rental venture with Oyo Hotels and Homes, in the latest setback for the loss-making Indian startup.",
    "Italian Prime Minister Giuseppe Conte welcomed the planned merger of Fiat Chrysler with its French rival PSA in an newspaper interview on Thursday, but said protecting jobs would be a top priority.",
    "Fiat Chrysler and Peugeot maker PSA face the challenge of winning over regulators and delivering on a pledge to slash costs without closing factories after sealing a binding deal to create the world's fourth biggest carmaker.",
    "Norwegian Air hopes to agree compensation from Boeing by year-end over the grounding of the 737 MAX, the airline's acting CEO said, as it counts the costs of having 18 of the aircraft grounded since March.",
    "China's Ant Financial, a fintech affiliate of e-commerce giant Alibaba Group Holding Ltd , has quietly acquired a sizable stake in a Vietnamese e-wallet eMonkey, people familiar with the matter said.",
    "Chinese automakers Great Wall Motor and Changan Automobile are accelerating plans to build cars in India after the initial success of rival SAIC Motor in one of the world's biggest markets, three sources said.",
    "China's Dongfeng Motor Group and Peugeot maker PSA are extending their business cooperation, despite the Chinese company reducing its stake in PSA to help smooth the French carmaker's merger with Fiat Chrysler Automobiles (FCA).",
    "Swatch will be blocked from selling some parts to watchmakers, ranging from Richemont's Cartier to Chopard and Breitling, from next year in an escalation of a row between the world's biggest watchmaker and Switzerland's competition regulator.",
    "British engineering company Rolls-Royce gave a first look at a one-seater electric aircraft on Thursday it hopes will fly in late Spring next year and become the world's fastest all-electric aircraft.",
    "The number of Americans filing applications for unemployment benefits dropped from more than a two-year high last week, pointing to sustained labor market strength.",
    "Facebook's Libra project has no solid plan yet for how or where it will be launched next year, a member of the board that will oversee the cryptocurrency told Reuters on Thursday.",
    "Citi Group and hedge fund ValueAct Capital said on Thursday they would extend an existing information-sharing agreement for another two years through the end of 2021.",
    "Adidas will start selling a new collection designed with singer Beyonce on Jan. 18 in a relaunch of her Ivy Park brand that includes shoes, clothes and accessories, mostly in maroon, orange and cream.",
    "China on Thursday unveiled a new list of import tariff exemptions for six chemical and oil products from the United States, days after the world's two largest economies announced a Phase 1 trade deal.",
    "Volkswagen has attracted bids from Europe's Innio, Japan's Mitsubishi Heavy and U.S.-based Cummins for its MAN Energy Solutions, which makes diesel engines for ships and power generators, people close to the matter said.",
    "U.S. President Donald Trump called Boeing Chief Executive Dennis Muilenburg this week to ask about the status of 737 MAX production, two people briefed on the matter confirmed.",
    "General Motors Co said on Thursday it is issuing recalls for more than 900,000 vehicles worldwide in two separate campaigns to address brake software issues and fire risks.",
    "U.S. Treasury Secretary Steven Mnuchin said on Thursday the United States and China would sign their so-called Phase one trade pact at the beginning of January, adding that it was completely finished and just undergoing a technical scrub.",
    "Accenture Plc beat Wall Street estimates for first-quarter earnings on Thursday, as investments in its fast-growing digital and cloud services businesses continue to pay off."
]

# --- Execution Loop ---
results = []

print(f"Processing {len(headlines)} headlines...")

# TQDM provides a progress bar for the loop
for h in tqdm(headlines):
    score = get_relevance_score(h)
    print(f"Headline: {h}\nScore: {score}\n")
    results.append({"headline": h, "score": score})

# Create DataFrame
df = pd.DataFrame(results)

# Display top 10
print("-" * 60)
print(df.head(10))

# Save to CSV (Optional)
# df.to_csv("scores_only.csv", index=False)
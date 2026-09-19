# Transcript: "wtf is jev?" — Syntax (CJ), YouTube QbYBRjOaGOo, 2026-09-17, ~18 min

Auto-generated captions via yt-dlp, lightly de-duplicated. Speaker: CJ (Syntax / Sentry).

What's up? CJ here. A new AI model has been released and we're going to talk about what it is, how 
it works, why you should care, and then show what some people have built with it, as well as show 
some demos that I've created that put it into action. So, this model is called Jev. It's from a 
company called Types AI, and it's headed by Dio Almata. So, they're known for co-inventing ChatGpt 
and RHF or reinforcement learning through human feedback. Um, so they've been working on AI for a 
while and they have released this model called Jeb. Now the headline is that it's very fast and 
it's very cheap. So it's 20 to 200 times faster than traditional LLMs and it's 40 to 400 times 
cheaper than traditional LLMs. And the thing to note about Jev is it is not an LLM. It's a 
classifier. It can be used in decision engines. And so let's let's talk about what that actually 
means. And so if you look at their launch post, this model Jev is part of what's called system one. 
And the term system one comes from a book called thinking fast and slow by Daniel Kenman. And he 
talks about how the brain works. And system one is known as the fast, automatic, frequent or 
emotional part of the brain that can like act quickly. And system two is more slow, effortful, 
infrequent, maybe requires some background processing. And so when we're thinking about AI models, 
Jev fits into system one. It's really fast. It answers questions immediately. And then traditional 
LLMs like Claude or ChatgBT or Gemini, those will fit into something like system 2. They can think 
for a while. They're a bit slower. And so those are known as system 2. So Jev is fast, it's system 
one, and it's not an LLM. And so the way that you work with Jev is you provide it your data. This 
is going to be like a customer email that you're trying to classify or some code that you're trying 
to analyze. Basically, this is your context. And then you also provide questions. What do you want 
to know about this piece of data? Now the data is known as state. And then your questions are going 
to be fixed options in your request. And then you pass that into Jev and it's going to respond with 
a probability for each of the options that you provided. And so the idea with Jev is you're very 
much pairing its output with your own more deterministic code that branches on the decisions that 
it's made based on your input and your questions. And in turn, you can perform some sort of action. 
Now there are three types of questions that Jev supports. You have yes or no questions which are 
known as nules. You also have pick one or options. This is known as a choice. And then you have 
rate on a scale and this is known as a score. So for example, you could provide a email that you 
got from a customer and ask Jev is this a refund request and it could answer yes or no with a 
certain probability for yes or no. Or you could say given this email which one of my teams should 
respond to it and you can give those teams as choices to Jev and then Jev will pick a team based on 
the email. And then similarly you could ask how frustrated is the user and then Jev could respond 
with a frustration score. Now the code for this is pretty straightforward. If you've ever talked to 
any sort of AI API using an SDK, it works exactly the same way. So you install Typesafe AI and then 
you talk to system one and you give it your state. So in this example, the state is maybe some 
email that we're trying to classify. So it says, "Hi, I was charged twice for my order. I'm really 
annoyed. Can I get a refund?" And then you specify which questions Jev should answer about that 
email. And so all of these are imported from the SDK. You have choice, null, and score. And then 
out the other side, you get an object that's strongly typed that has all of those properties that 
you can now branch on and write code to actually do something based on how Jev classified this 
particular email. Now, the way that this is different from how agents are written today is 
typically if you're building out an agent, you have just some giant prompt, right? So, you receive 
some user email that you want to classify and you have a bunch of questions that you basically put 
into your LLM prompt. Like, what's the intent of this message? Is it safe? Does it need a human? 
What tool should I call? Do I support answering this type of question? And then you tell the LLM 
based on all of those things, write a reply to the message and then and then send it back. And so, 
this is very prone to error. And so if you've ever written an agent running in production, you have 
these finely crafted prompts that you're always tweaking and trying to tweak the wording just to 
get the LLM to respond in the way that you would like. And there's a very high error rate. And so 
this typically would be written out using something like the AI SDK from Versell where you're 
talking to a specific model. You give it instructions and then provide the input. And modern LLMs 
and SDKs do support what's known as structured output. Right? So you can tell an LLM that it should 
respond with an object that aderes to this schema, but it doesn't always do that, right? You 
typically have to wrap this in a try catch because sometimes the AI's answer get cuts off or maybe 
sometimes it'll hallucinate properties. And so while this works for a lot of cases, it's very 
errorprone. And so the same thing with Jev is very simple, right? You pass in the state, ask it the 
questions, and then you are always going to get back the specific answer to those things. And it's 
going to be much faster than waiting for a response from an LLM. And so the way you can think about 
Jev is kind of like the first thing in the line of duty when it's trying to respond in these agent 
workflows, right? So you have some sort of incoming message. You make one really fast call to Jev 
to classify that message and then act on the answers using your own code that you've written. And 
so if a user, for instance, is asking for order status, you can immediately do a tool call and look 
it up. No LLM required. If they're asking a specific question about a product, maybe then you call 
out to an LLM to generate an answer based on their question. Or if the user is angry or unclear, 
maybe you can then triage that so an actual human can respond. Or if the user's message is trying 
to do something malicious like a jailbreak, you could then immediately block that message. So 
that's what Jev is. It's really fast, it's really cheap, and it allows you to write more 
deterministic code rather than relying on the output of an LLM. So let's look at some examples of 
what people actually have built with it. And this example is a browser use. So this user asked Jev 
to find flights for a particular destination and it only took 7 seconds. So they basically hooked 
up Jev to be able to see snapshots of the browser to decide what should be filled in and were able 
to book a flight within 7 seconds. And if you've ever used like the Playright MCP or like any sort 
of agent browser, you know that these things are typically slow, right? There's a lot of back and 
forth with the LLM. And so if you have more deterministic workflows where you potentially know the 
state of the page and what it should be looking for, you can write really fast workflows with Jev. 
You can also do classification. So all of the examples we've used so far are classifying like 
support emails. But you might have a lot of data at your company that you're trying to classify. 
And so this person classified over a thousand AI research papers to determine what category do they 
fit into so they could better uh read them later on. And for classifying all of them, it cost 8 
cents and was 256 milliseconds per classification. So there's a lot of classifying type of 
workflows that people have been dependent on LLMs to do, but now you could reach for Jev and do it 
cheaper and faster. I keep saying cheaper and faster. That that's basically the headline here. 
Also, uh, JMED, who's actually the creator of Hiring Cafe, used Jev to classify is a given resume a 
good fit for a job posting. And you can see they have their accuracy scores and costs. So comparing 
it to some of these other smaller LLMs to do this type of classification, it's 10 times cheaper. 
And another classification of course is emails. But what about just your personal inbox? And so 
this was an example from Vogle where he had Jev classify the priority whether or not an email was 
spam. Should I reply to this email? And you can actually see in real time how fast Jev is 
classifying each one of those emails. So it's it's fast. It's it's blazing fast. Now the other 
category of using Jev is more like real-time interaction. And so this is an example where Jev is 
classifying the text as it's being written in real time. So it's it's pulling out the tone, the 
conviction, the urgency. And so you can imagine having a feature like this inside of like a word 
processor or a place where you're writing documents and in real time you're getting all of that 
feedback about the stuff that you're writing. And you could also have deterministic rules. So you 
see they have a score here for reads AI written. We know a lot of the signs of AI writing, right? 
it it says like it's this, it's not that, or it has sentences that are like three words long. And 
so you could write those as more deterministic rules that it can then use to determine is it AI 
written. And so instead of using some pre-built tool that can do classification, you essentially 
come up with all of those rules yourself. And so I could see schools using this for like checking 
for copyright, checking for is it AI, checking for duplication, that kind of thing. And again, 
because it's fast, you can do this kind of thing really quickly and really cheaply. Now, another 
type of real-time interaction is a browser extension. So they built an X browser extension that 
automatically hides posts that it thinks could be rage bait or talking about crypto or political 
discussion. So you can essentially in real time fix your algorithm. I can imagine building a 
similar extension for YouTube that hides certain results or certain recommendations based on your 
preferences. And another really great example of real-time analysis is something like a debate. 
Now, I'm not going to get political here, but imagine analyzing the words that people are saying 
real time in a debate and being able to fact check those and show it in real time on the screen. 
Now, we've seen demos like this built with LLMs, but Jeb is fast and so it can actually analyze 
every single word, every single sentence that someone is saying and in real time give you a BS 
meter of of the people speaking. And another type of real-time example is games. So this person 
actually coded the rules of Tetris into Jev and Jev is deciding what to do. Should it press down? 
Should it rotate the shape? And it's doing all of those as API calls in real time. So it's fast 
enough to be able to play a game like this. And another real-time example is actually self-driving. 
So they hook Jev up to a driving simulator where you have all of the the things that a a 
self-driving car would know about like is there a stop sign, is there traffic, are there 
pedestrians, is there a crosswalk? And in real time, it's making decisions about what it should do, 
like should it accelerate, should it stop, how much should it turn, and it's doing all of that in 
real time. And finally, you can do things like code analysis. So here they've written essentially a 
Jev code review. So they've decided all the things they care about in terms of analyzing a code 
review, like maybe the commit messages, uh the complexity of the code, these different types of 
things. And then they have a risk matrix that can score every modified file in terms of like is 
there a security issue or is this bad coding practice? Those types of things. And so there's a ton 
of things you can do with Jev, but I'm going to show you a few more examples of what we're doing 
here at syntax and also at Sentry. So on the syntax website for every podcast, we actually have 
this giant LLM prompt that summarizes the episode based on the transcript. And you can see we have 
lots of little rules written in here. And so in the transcript, if it ever detects the word sentry, 
typically it uses this spelling instead of this spelling. And we say sentry a lot because 
everything here at syntax is brought to you by century. So we have fixed rules that we pass to an 
LLM that say, "Hey, replace all of these occurrences." And then similarly, it hears the word syntax 
wrong. And so we we have to come up with all these spellings that we tell an LLM that it should uh 
replace whenever it's doing the the summary there. And so big long prompt, lots of rules here. This 
is a perfect use case for something like Jev. And the workflow is essentially that audio gets 
transcribed using deepgram. We then write notes using an LLM. So this basically gives you the 
summary of that episode. And then what we could do with Jev is also verify the claims that the LL 
LLM is output. So we extract topics, tags, chapter markers, and we could essentially take the 
output from the LLM and run Jeb through every single claim and make sure that that it's truthful 
and didn't hallucinate anything. And so in the code, we can basically do the same thing. pass it to 
the AI and then after that do a follow-up with Jev to verify every single one of the claims. Now, 
another example is a model router. This is a very common use case when you're building out agents 
and this is Junior, which is Century's Slackbot. Essentially, it can answer questions, open GitHub 
issues, write code in the background, pull from linear, pull from notion. It's basically hooked up 
to everything at Sentry. And when it's responding, it actually chooses which model it should 
respond with. And so we have this entire turn router here with a prompt that describes what types 
of requests should go to what models. And again, this is a perfect use case for Jev. So this can 
basically be turned into two questions for Jev. First, how much reasoning is required? And then 
which profile does it fit? And each profile specifies a certain model that it should talk to. So 
before ever reaching for an LLM, we first classify the message and then we can pass it off to an 
LLM. Now, finally, I want to show you the demo that I built, which is essentially a chatbot that 
does not use an LLM at all. It only used tool calls. And so, I actually used the smart home 
assistant demo from Typesafe AI. It's in their docs as a starting point. And we have this chatbot 
here that does not use an LLM. So, for instance, I can ask it, "What's the weather in Denver 
tomorrow?" And the interesting thing about this is we can basically take in plain text. So, there's 
certain types of calls that we do to Jev to determine what is the user actually asking. So it 
doesn't always have to be in this format. I could even say like tell me the weather in Denver, 
please. Right? So it doesn't have to be a specific question. And that's the interesting thing about 
how you can use Jev. We essentially ask Jev, what is the user asking? What's the topic of the 
question? What tool should I call based on their question? And then we're able to get back a 
response. So no LLM is involved here. And on the right hand side, you can see the actual flow. So 
first it asks Jev, what is the user requesting? So is this a request? Is this a confirmation or did 
the user cancel something? It said this is a request. Then it said which tool would fulfill the 
user's last message. And we have lots of different tools registered. So I have like a units tool, a 
web search tool, a Wikipedia tool, a recipe search tool, a to-do API tool. Also I hooked up home 
assistant to it. So basically we present Jev with all of these tools and say which one do you think 
we should call? It chose that one. We also have another question in here that says for a weather 
request which city or place does the user want the weather for? And this uses a piece of jev that 
can actually extract out from the input state. And so this is how you can actually use the user's 
message and include that in the response. So Jev is smart enough to pick out the word Denver and 
provide probabilities for all of what the user could have been asking for. And then it also tries 
to pick the time like is it now, is it today, is it tomorrow? And then also picks the units. And 
then based on all of that, we get back a response that says what tool to call, what input to use 
and then we make that deterministic tool call in our code. And the other cool thing is we can also 
do follow-ups, right? So if I say what's the high in Celsius, we're actually including the 
conversation history in its context. So it can actually know that, oh, I got back the response of 
86° and so I'm going to convert that. And also instead of just guessing or trying to calculate the 
output, it made a call to convert units. So we have a built-in tool that can convert between 
different systems of measure and it deterministically called that tool as well. Now the other tool 
that I hooked this up to is Wikipedia. And so I can ask it questions like how tall is Mount Reneer 
and it will know to reach out to Wikipedia and then there's a way to ask Jev to point to a specific 
line in the state about what answers a particular question and it was able to do that. So it pulled 
the Wikipedia article and then quoted the specific sentence that said how tall Mount Reneer is. And 
so we can ask it all kinds of questions like uh when did the first Twilight movie come out? And in 
this case, it actually used our web search instead of Wikipedia and it knows that the answer is 
2008. So this isn't that impressive because we we've been able to do this with LLMs. But the fact 
that you can essentially take all of these deterministic tools that we've already created because 
we've been using them as MCPs with LLMs and then craft the answers in a way so that Jev can 
actually respond to them without generating any text. To me, this is the ideal use case for like 
factual responses. Right? Right now when we talk to LLMs, it just poops out a bunch of text and you 
have no idea where that text came from. But if you have a chatbot that's more grounded and 
basically uses tool calls for every request instead instead of potentially hallucinating the 
response, you can get much more grounded answers. And so that's pretty slick. Now, the last example 
I have here is with my smart home. So I hook this up to the Home Assistant MCP and I actually have 
a light back there that is controlled by Home Assistant. So I can ask it to turn it on and off. So 
I'll say turn off the top bulb instant. And so there's no there's no movie magic here. This is the 
other cool thing about how fast this is. Like if I were to ask an LLM this, it might respond with a 
tool call and then have to call the tool call. This took 300 milliseconds to ask Jev to answer the 
question and then call the the particular home assistant API endpoint to actually turn the thing 
off. Now I can also say uh turn it back on. It should turn on and I can say increase bright 
increase brightness to 100%. and it gets brighter. I can also say change to green. Boom. So again, 
it's not that impressive, but the fact that these responses are so fast and it's actually making 
decisions about my plain language text is is absolutely fascinating. Now, the way all this works is 
I basically just took existing MCP tools. I didn't modify them in any way and then just made it so 
that Jev can choose which tool it should potentially call. And as I showed in the Jev inspector, 
Jev responds with all of that in a single response. So it says what tool should I call? Uh what 
value should I send it? All of that. And the other reason this is so fast is all of that comes back 
in the same response. I don't have to prompt it multiple times like what tool should I call? What 
arguments should I invoke the tool with? It responds with all of that in like 200 milliseconds or 
less. So that's what Jev is. That's how people have been using it. That's how I've been using it. 
Definitely check it out. If you go to their docs, they have all of this stuff explained as well as 
various patterns you can use and also cookbooks. There's just so many possibilities. Definitely 
give it a try. Right now, they do have a weight list. So, if you sign up on the wait list, I got in 
within a day of requesting the weight list, but if you don't want to wait on the wait list, it's 
actually available via the Verscell AI gateway. It might be slightly more expensive, but still, you 
can basically use your existing plan and talk to Jeb in the exact same way. So, thank you so much 
for watching this video. So, if you have any questions about what I built, let me know down in the 
comments. If you've been building things with Jev, let us know down in the comments as well.
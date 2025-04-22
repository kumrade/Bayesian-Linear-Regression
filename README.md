Bayesian Linear Regression
In regular (frequentist) linear regression, we calculate the best-fitting line through the data just once, using the entire dataset. But in Bayesian regression, we start with a belief, update it step by step as we get more data, and keep refining it. It's like learning from experience, one piece at a time.

📚 Bayesian vs Frequentist
Frequentist approach: Just crunch the numbers and get one fixed answer.

Bayesian approach: Start with a belief → get new data → update belief → repeat.

Imagine you're drawing a line through points. At first, you're guessing. But as more points show up, your guess gets better and better!

🔁 How Bayesian Updating Works
We use Bayes' Theorem, which in simple terms says:

Your updated belief (posterior) = your starting belief (prior) × new evidence (likelihood)

Every time you see new data, you shift your belief a little — slowly shaping it toward the truth.

🛠️ The Example in the Code
Let’s say you have 500 points from a noisy straight line (like before).

You start with a guess — say the line is flat (horizontal).

Then you look at the first two data points, and update your guess.

Then you take the next two points and update again.

Do this 250 times (500 points = 250 pairs).

Your belief about the line slowly changes — and gets closer and closer to the true line.

This is called recursive Bayesian learning. The more data you see, the more confident you become.

🎨 What’s the Code Doing?
Starts with a belief that the line has intercept = 0.5, slope = 0.5

Goes through 250 data pairs, updating the intercept and slope each time

Plots the line at different stages:

At the beginning (initial belief)

After 1st update

After 10 updates

After 100 updates

And finally, the actual line (true values)

📊 What does the plot show?
You’ll see the line:

Starting off wrong (maybe flat or tilted),

Then slowly adjusting,

And finally becoming very close to the true line (which had intercept = 1.0, slope = 2.0)

🤖 Modern Tools
While this was all done manually, these days you can use tools like PyMC3 to handle all the math for Bayesian regression. But it's still good to know what’s going on behind the scenes!

🧁 In a Nutshell:
Bayesian regression is like learning in real-time. You start with a guess, and then each new observation helps you fine-tune your understanding. Over time, you get closer to the truth, even if you started off with a wrong assumption.

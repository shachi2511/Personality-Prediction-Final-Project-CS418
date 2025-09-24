# Team Reflection

This document is a collective reflection on your team’s final project experience. Discuss what you learned, what went well, what was challenging, and what you'd do differently next time. This is not just for grading — it's for your own growth as future data scientists and collaborators.

---

## 1. What Did You Learn?

That it is hard to 100% determine through text about the personality. We can find it through using Ai, but overall, using Ml as a test would be tough.
We also got to learn complex topics such as vector embedding, usage of knowledge distillation and using models like "all-MiniLM-L6-v2" and BERT to implement the code.

---

## 2. What Went Well?

- We were able to get a model to get up to 78% accuracy rate. 
- The tasks were very evenly dedicated

---

## 3.  What Was Challenging?

What obstacles did you face — technically or as a team? How did you address them?

- The First issue was to find a proper dataset containing social media posts of individuals to analyse — went through a number of research papers and found them on git.
- Deciding on an approach — the first approach based on frequency proved to be extremely random and unfocused on the sentiment of the text, so we changed to a second approach which didn't give us a proper accuracy rate due to which we had to decide on the final approach.
- Finding a good model which can predict personality trait by just looking at the social media comments, captions or conversations is really complex. Needed a good attention-based mechanism that not only considers the frequency but also the nature of the sentences individually and then doing for a collection of same sentences per user. ~**Shubham**
---

## 4. What Would You Do Differently?

If you could restart this project, what would you change or improve?

- Analyse the model's pros and cons completely before implementing it and going through all the possible ways before deciding on one.
- Take a different dataset or find a better way to handle biasedness towards the majority label (ie introvert).  ~**Shubham**
- use a good GPU and a better pretained model than the one being used. ~**Shubham**
---

## 5. Final Thoughts

Any final takeaways, surprises, or things you’re proud of?

- we have gone through various ways to analyse a person's personality through their conversations. This has made us more aware socially.
- something we are proud of is the recall value of our model for introverts was about 97.4% making it very reliable to predict introverts through their social media presence.
- what seemed really simple turned out to be very complicated task - predicting user's personality basis on social media texts. ~**Shubham**
- learnt something new about teacher student approach (knowledge distillation), how it can tackle really complex tasks. ~**Shubham**

---

Note: **Each team member should contribute to this reflection.** 

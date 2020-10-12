# ECE4078 Practical Week 09

This week we will cover Model-Free and Deep Reinforcement Learning

## Coding Exercise

You are tasked with extending the DQN implementation seen during the lecture so it includes one target and one policy network.


## Submission

- You should submit a copy of this week's notebook (with your solutions) and your trained network by Monday 11:55pm at the latest. Submissions will be done through Moodle.
- Your submision must follow this naming convention: [Student_ID]_Practical09.ipynb and [Student_ID]_DQN.pt
- Set the **RUN_TRAINING** flag to False before submitting your notebook</b></p> 
- Remove all print statements from your code
- Make sure that you do not change the name of the DQN class
- You can tune the hyper-parameters if needed or modify the architecture of the DQN approximator. During grading, we should be able to create a copy of your DQN network using the submitted file without problem. Code for testing this is provided in the notebook


## Marking:

- All programming exercises will be graded automatically. Please make sure to not change the names of the functions you are asked to complete. Verify that each function returns the expected value type and format.
- You will be graded based on the performance of your network. Your solution will be executed for a total of 100 trials and the average return will be used to determine your grade. The grading scale is:

| Avg. Return | Marks       |
| ----------- | ----------- |
| < 100       | 1  pt       |
| 101 - 120   | 2  pts      |
| 121 - 140   | 3  pts      |
| 141 - 160   | 4  pts      |
| 161 - 180   | 5  pts      |
| > 180       | 6  pts      |
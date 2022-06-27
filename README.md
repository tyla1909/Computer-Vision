# Computer-Vision
Using Computer Vision to create the classic Rock, Paper, Scissors game.
This is a computer vision project. I used Teachable-Machine to create a model with four different classes, namely; Rock, Papers, Scissors and Nothing. Each class was trained with images of myself showing each option to the camera. I created a new virtual environment for this project. In the new environment i installed the necessary requirements for the project such as opencv-python, tensorflow, and ipykernel. 
I started by creating a RPS_GAME Class, and added the different attributes needed. I defined my user_move function which returns the output of the model trained, the get_prediction function returns the user choice and the get_computer_choice function returns the computer choice among the options(Rock,Paper,Scissors) given using random.choice method.
The logic of the game was done using if-elif and else statements.
The getting_started function was defined using the cv2 module and the time function was also used to program the countdown.
The get_winner function returns the winner in each round of the game while the number of wins for each player is less than three wins

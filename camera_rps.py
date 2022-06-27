import random
import time
import cv2
from keras.models import load_model
import numpy as np
class RPS_Game():     
      def __init__(self):
            self.name = ''
            self.user_choice = ''
            self.computer_choice = ''
            self.computer_wins = 0
            self.user_wins = 0
            self.options = ['Rock', 'Paper','Scissors']
            self.rounds = 1

      def user_move(self):
         if prediction[0][0] > 0.5:
            user_choice = 'Rock'
            return user_choice
         elif prediction[0][1] > 0.5:
            user_choice = 'Paper'
            return user_choice
         elif prediction[0][2] > 0.5:
            user_choice = 'Scissors'
            return user_choice
         else:
            user_choice = 'Nothing'
            return user_choice

     

      def get_prediction(self, frame, data, model):
            ret,frame = cap.read()
            resized_frame = cv2.resize(frame, (224, 224), interpolation = cv2.INTER_AREA)
            image_np = np.array(resized_frame)
            normalized_image = (image_np.astype(np.float32) / 127.0) - 1 # Normalize the image
            data[0] = normalized_image
            prediction = model.predict(data)
            prediction = np.argmax(prediction[0])
            self.user_choice = self.user_move()
            if self.user_choice != 'Nothing' and self.user_choice != 'None':
                  ret, frame = cap.read()
                  self.show_text(frame, f"User's Choice: {self.user_choice}", (50,75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_4)
                  cv2.waitKey(500)
                  cv2.imshow('RPS GAME BY TYLA PETERS', frame)
                  self.get_computer_choice()
            else:
                self.getting_started()

      def getting_started(self):
            pressKey = cv2.waitKey(1) & 0xFF
            #Starting the game
            if self.user_wins == 3 or self.computer_wins == 3:
                if self.user_wins == 3:
                  self.show_text(frame,f'USER WINS THE GAME,  Press P to play again or Q to end game', (100, 225),  cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
                elif self.computer_wins == 3:
                  self.show_text(frame,f'COMPUTER WINS THE GAME,  Press P to play again or Q to end game', (100, 225),  cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
                if pressKey == ord('q'):
                    self.computer_wins = 0
                    self.user_wins = 0
                    self.show_text(frame, 'GAME OVER', (200, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_4)
                    cv2.waitKey(250)
                    # After the loop release the cap object
                    cap.release()
                    # Destroy all the windows
                    cv2.destroyAllWindows()
                elif pressKey == ord('p'):
                    self.computer_wins = 0
                    self.user_wins = 0
                    self.getting_started()

            elif self.user_wins == 0 and self.computer_wins == 0:
                self.show_text(frame, 'Press s to start or q to quit', (100, 225),  cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2, cv2.LINE_AA)
                if pressKey == ord('q'):
                        self.show_text(frame, 'GAME OVER', (200, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_4)
                        cv2.waitKey(250)
                        # After the loop release the cap object
                        cap.release()
                        # Destroy all the windows
                        cv2.destroyAllWindows()

                elif pressKey == ord('s'):
                              previous = time.time()
                              countdown  = 5
                              while countdown > 0:
                                    self.show_text(frame, f'SHOW YOUR MOVE TO THE CAMERA:{countdown}',(50, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)         
                                    cv2.waitKey(125)


                                    
                                    current = time.time()
                                    if current - previous >= 1:
                                          previous = current
                                          countdown = countdown - 1
                        
                              else:                           
                                    self.get_prediction(frame, data, model)

            

            else:
            
                  self.show_text(cap, 'Press c to start next round or q to quit', (25, 250),  cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
                  if pressKey == ord('q'):
                        self.show_text(frame, 'GAME OVER', (200, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_4)
                        cv2.waitKey(250)
                        # After the loop release the cap object
                        cap.release()
                        # Destroy all the windows
                        cv2.destroyAllWindows()

                  elif pressKey == ord('c'):
                        previous = time.time()
                        countdown  = 5
                        while countdown > 0:
                              self.show_text(frame, f'SHOW YOUR MOVE TO THE CAMERA:{countdown}',(50, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)        
                              cv2.waitKey(125)


                              
                              current = time.time()
                              if current - previous >= 1:
                                    previous = current
                                    countdown = countdown - 1

                        else: 
                              self.get_prediction(frame, data, model)
      def get_computer_choice(self):
            ret, frame = cap.read()
            cv2.rectangle(frame, (800, 100), (1200, 500), (255, 255, 255), 2)
            self.computer_choice = random.choice(self.options)
            self.show_text(frame, f"Computer's Choice: {self.computer_choice}", (400,75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2, cv2.LINE_4)
            cv2.waitKey(500)
            cv2.imshow('RPS GAME BY TYLA PETERS', frame)
            self.get_winner()
      def get_winner(self):
            if self.user_wins < 3 and self.computer_wins < 3 :
                 if self.computer_choice == self.user_choice:
                        self.show_text(frame, f"Round {self.rounds} is a Tie ", (100, 300), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 2, cv2.LINE_8)
                        cv2.waitKey(2500)
                        self.play_again()
                        self.rounds +=1

                 elif self.user_choice == ('Rock'):
                        if self.computer_choice == 'Paper':
                              self.computer_wins += 1
                              self.show_text(frame, f"Computer Wins Round {self.rounds}!", (75, 250), cv2.FONT_HERSHEY_SIMPLEX, 1.7, (0, 255, 255), 2, cv2.LINE_8) 
                              cv2.waitKey(2500)
                              self.play_again()
                              self.rounds +=1
                        elif self.computer_choice == 'Scissors':
                              self.user_wins +=1
                              self.show_text(frame, f"User Wins Round {self.rounds}", (100, 250), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 2, cv2.LINE_8)
                              cv2.waitKey(2500)
                              self.play_again()
                              self.rounds +=1
                 elif self.user_choice == 'Paper':
                        if self.computer_choice == 'Scissors':
                              self.computer_wins += 1
                              self.show_text(frame, f"Computer Wins Round {self.rounds}", (75, 250), cv2.FONT_HERSHEY_SIMPLEX, 1.7, (0, 255, 255), 2, cv2.LINE_8)  
                              cv2.waitKey(2500)
                              self.play_again()
                              self.rounds +=1
                        elif self.computer_choice == 'Rock':
                              self.user_wins += 1
                              self.show_text(frame, f"User Wins Round {self.rounds}", (100, 250), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 2, cv2.LINE_8) 
                              cv2.waitKey(2500)
                              self.play_again()
                              self.rounds +=1
                 elif self.user_choice == 'Scissors':
                        if self.computer_choice == 'Paper':
                              self.user_wins += 1
                              self.show_text(frame, f"User Wins Round {self.rounds}", (100, 250), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 2, cv2.LINE_8)
                              cv2.waitKey(2500)
                              self.play_again()
                              self.rounds +=1
                        elif self.computer_choice == 'Rock':
                              self.computer_wins += 1
                              self.show_text(frame, f"Computer Wins Round {self.rounds}", (75, 250), cv2.FONT_HERSHEY_SIMPLEX, 1.7, (0, 255, 255), 2, cv2.LINE_8)
                              cv2.waitKey(2500)
                              self.play_again()
                              self.rounds +=1    
      def play_again(self):
            if self.computer_wins == 3 or self.user_wins == 3:
                  
                  if self.computer_wins == 3:
                        cv2.putText(frame, f"COMPUTER WINS THE GAME", (50, 250), cv2.FONT_HERSHEY_SIMPLEX, 1.7, (255, 255, 255), 2, cv2.LINE_AA)
                        cv2.waitKey(2500)
                        
            
                  elif self.user_wins == 3:
                        cv2.putText(frame, f'USER WINS THE GAME', (50, 250), cv2.FONT_HERSHEY_SIMPLEX, 1.7, (255, 255, 255), 2, cv2.LINE_AA)
                        cv2.waitKey(2500)
                                                                                                                                                                                                                                                                                                                                                                                                                                                                              
                        
            else:
                                                                                                                                                                                                                                                
                  self.getting_started()
      def show_text(self, frame, text, text_position, font, fontsize, color, thickness, linetype):
            ret, frame = cap.read()
            cv2.putText(frame, text, text_position, font, fontsize, (0, 255, 255),thickness,cv2.LINE_AA)
            cv2.putText(frame, f'User Wins:' + str(self.user_wins), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 0, cv2.LINE_4)
            cv2.putText(frame, f'Computer Wins:' + str(self.computer_wins), (400, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 0, cv2.LINE_4)
            cv2.putText(frame, f'Rounds:' + str(self.rounds), (250, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_4)
            cv2.putText(frame, f"User's choice: {self.user_choice}", (50,75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_4)
            cv2.putText(frame, f"Computer's choice: {self.computer_choice}", (400,75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_4)
            cv2.imshow('RPS GAME BY TYLA PETERS', frame)
model = load_model('keras_model.h5')
cap = cv2.VideoCapture(0)
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
play_RPS_GAME = RPS_Game()


while True: 
      ret, frame = cap.read()
      resized_frame = cv2.resize(frame, (224, 224), interpolation = cv2.INTER_AREA)
      image_np = np.array(resized_frame)
      normalized_image = (image_np.astype(np.float32) / 127.0) - 1 # Normalize the image
      data[0] = normalized_image
      prediction = model.predict(data)
      cv2.imshow('RPS GAME BY TYLA PETERS', frame)
      play_RPS_GAME.getting_started()
      if cv2.waitKey(1) & 0xFF == ord('q'):
            break
                        
cap.release()
cv2.destroyAllWindows()
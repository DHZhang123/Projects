import random


class Pet:
    """A virtual pet that can be fed, played with, and bathed. Multiple conditionsincluding name, age, happiness, etc."""
    def __init__(self, name:str) -> None:
        """Initializes a new instance of the Pet class with the given name."""

        self.name = name
        self.fullness = 8
        self.happiness = 8
        self.cleanliness = 8
        self.alive = True
        self.stage = "egg"
        self.progress = 1


    def feed(self) -> None:
        """Feeds by 3 and adjust cleanliness and fullness accordingly."""
        if self.fullness == 10:
        #If fullness is already at max, cleanliness goes down
            self.cleanliness = max(self.cleanliness - 2, 1)
        #Otherwise increase fullness to a max of 10
        else:
            self.fullness = min(self.fullness + 3, 10)


    def play(self) -> None:
        """Increases happiness and adjust other conditions accordingly."""
        if self.happiness == 10:
        #If happiness is max, fullness goes down
            self.fullness = max(self.fullness - 2, 1)
        #Otherwise increase happiness to a max of 10
        else:
            self.happiness = min(self.happiness + 3, 10)


    def bathe(self) -> None:
        """Increase cleaniliness and adjsut accordingly."""
        if self.cleanliness == 10:
        #If cleanliness os max, decrease happiness
            self.happiness = max(self.happiness - 2, 1)
        #Otherwise increase cleanliness
        else:
            self.cleanliness = min(self.cleanliness + 3, 10)

    def age_up(self) -> None:
        """Change the value of stage to the nect."""
        #Depending on the current age, change to the next
        if self.stage == "egg":
            self.stage = "baby"
        elif self.stage == "baby":
            self.stage = "child"
        elif self.stage == "child":
            self.stage = "adult"
        #Reset progress
        self.progress = 1


    def status(self) -> str:
        """Based on the values of fullness, cleanliness, and happiness, return status."""
        #If any conditions are 1, pet is dead
        if self.fullness == 1 or self.happiness == 1 or self.cleanliness == 1:
            self.alive = Falss
            return "dead"
        #If values are all above 5, pet is fine
        elif self.fullness > 5 and self.happiness > 5 and self.cleanliness > 5:
            return "fine"
        #In between 1 and 5 means distressed
        else:
            return "distress"


    def time_step(self) -> str:
        """Randomly decreases one condition and increase progress by one."""
        #Use random to select one condition
        choice = random.choice(["fullness", "happiness", "cleanliness"])
        #Depending on which condtion is picked, decrease that value by 1
        if choice == "fullness":
            self.fullness -= 1
        elif choice == "happiness":
            self.happiness -= 1
        else:
            self.cleanliness -= 1
        #Increase progress by one, if progress = 20, call age_up
        self.progress += 1
        if self.progress == 20:
            self.age_up()
        return self.status()



import turtle
import time

def fill_circle(turtle, color, radius, position):
        turtle.up()
        turtle.goto(position)
        turtle.down()
        turtle.fillcolor(color)
        turtle.begin_fill()
        turtle.circle(radius)
        turtle.end_fill()
        turtle.up()
        turtle.goto(0,0)
        
class TamagotchiGame:
    def __init__(self, name: str):
        """Creates a Tamagotchi Pet with the given name"""
        turtle.tracer(0,0)
        self.pet = Pet(name)
        self.pen = turtle.Turtle()
        self.pen.up()
        self.pen.hideturtle()

    def draw_egg(self):
        fill_circle(self.pen, "green", 20, (0,0))
        fill_circle(self.pen, "white", 5, (10,20))
        fill_circle(self.pen, "white", 5, (-10,20))
        fill_circle(self.pen, "black", 2, (10,22))
        fill_circle(self.pen, "black", 2, (-10,22))
        turtle.update()

    def draw_baby(self):
        fill_circle(self.pen, "red", 10, (15,0))
        fill_circle(self.pen, "red", 10, (-15,0))
        fill_circle(self.pen, "red", 30, (0,0))
        fill_circle(self.pen, "white", 8, (15,30))
        fill_circle(self.pen, "white", 8, (-15,30))
        fill_circle(self.pen, "black", 4, (15,34))
        fill_circle(self.pen, "black", 4, (-15,34))
        turtle.update()

    def draw_child(self):
        fill_circle(self.pen, "purple", 14, (20,0))
        fill_circle(self.pen, "purple", 14, (-20,0))
        fill_circle(self.pen, "purple", 10, (40,40))
        fill_circle(self.pen, "purple", 10, (-40,40))
        fill_circle(self.pen, "purple", 40, (0,0))
        fill_circle(self.pen, "white", 10, (15,40))
        fill_circle(self.pen, "white", 10, (-15,40))
        fill_circle(self.pen, "black", 5, (15,44))
        fill_circle(self.pen, "black", 5, (-15,44))
        turtle.update()

    def draw_adult(self):
        fill_circle(self.pen, "blue", 18, (25,0))
        fill_circle(self.pen, "blue", 18, (-25,0))
        fill_circle(self.pen, "blue", 12, (50,50))
        fill_circle(self.pen, "blue", 12, (-50,50))
        fill_circle(self.pen, "blue", 50, (0,0))
        fill_circle(self.pen, "purple", 5, (0,35))
        fill_circle(self.pen, "white", 12, (15,50))
        fill_circle(self.pen, "white", 12, (-15,50))
        fill_circle(self.pen, "black", 6, (15,55))
        fill_circle(self.pen, "black", 6, (-15,55))
        turtle.update()

    def draw_tombstone(self):
        self.pen.fillcolor("gray")
        self.pen.begin_fill()
        self.pen.forward(50)
        for i in range(2):
            self.pen.left(90)
            self.pen.forward(200)
            self.pen.left(90)
            self.pen.forward(100)
        self.pen.end_fill()
        self.pen.up()
        self.pen.goto(0,160)
        self.pen.write("RIP", align = "center", font=("Arial", 20, "normal"))
        self.pen.goto(0,140)
        self.pen.write(self.pet.name, align = "center", font=("Arial", 15, "normal"))
        self.pen.goto(0,0)
        turtle.update()

    def display(self):
        self.pen.clear()
        self.pen.up()
        self.pen.goto(0,-30)
        self.pen.write(self.pet.name, align = "center", font=("Arial", 20, "normal"))
        turtle.update()
        self.pen.goto(0,0)
        if self.pet.stage == "egg":
                self.draw_egg()
        elif self.pet.stage == "baby":
                self.draw_baby()
        elif self.pet.stage == "child":
                self.draw_child()
        else:
                self.draw_adult()
        if self.pet.status() == "distress":
                for i in range(2,5):
                        self.pen.goto(0,30*i)
                        self.pen.write("WWAHHHH!! :(", align = "center", font=("Arial", 30, "normal"))
                self.pen.goto(0,0)
                turtle.update()

    def feed(self):
        self.pet.feed()
        self.display()
        self.pen.goto(0,20)
        self.pen.write("NOM NOM NOM", align = "center", font=("Arial", 30, "normal"))
        turtle.update()
        self.pen.goto(0,0)
        time.sleep(2)
        self.display()

    def play(self):
        self.pet.play()
        self.display()
        self.pen.goto(0,20)
        self.pen.write("WEEEEE!!!!!", align = "center", font=("Arial", 30, "normal"))
        turtle.update()
        self.pen.goto(0,0)
        time.sleep(2)
        self.display()

    def bathe(self):
        self.pet.bathe()
        self.display()
        self.pen.goto(0,20)
        self.pen.write("SCRUB SCRUB SCRUB", align = "center", font=("Arial", 30, "normal"))
        turtle.update()
        self.pen.goto(0,0)
        time.sleep(2)
        self.display()

    def run(self) -> None:
        """Runs the Tamagotchi game"""
        self.display()
        time.sleep(2)
        state = self.pet.time_step()
        while state != "dead":
            self.display()
            self.pen.goto(0,-50)
            self.pen.write("Type 1 to feed, 2 to play, 3 to bathe", align = "center", font=("Arial", 15, "normal"))
            turtle.update()
            self.pen.goto(0,0)
            turtle.listen()
            turtle.onkey(self.feed, "1")
            turtle.onkey(self.play, "2")
            turtle.onkey(self.bathe, "3")
            time.sleep(1)
            state = self.pet.time_step()
        self.pen.clear()
        self.draw_tombstone()
        turtle.exitonclick()


            
        
def play_tamagotchi() -> None:
    """Runs the Tamagotchigame."""
    #Get a pet tamagotchi name
    pet_name = input("What would you like to name your Tamagotchi?")
    #Create an instance of the TamagotchiGame class
    game = TamagotchiGame(pet_name)
    #Run the game
    game.run()

    
        
        

            

    


    
        

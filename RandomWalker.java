import java.util.Random;

public class RandomWalker {
///write a program that simulates a drunkards random walk with the possibility of 4 directions
	
	public static void main(String[] args) {
		int N = Integer.parseInt(args[0]);
		int x = 0;
		int y = 0;
		int squared_dist = 0;
		///initialize int N, x, y, and squared_dist
		
		Random rand = new Random();
		///set Random to rand
		
		for (int i = 0; i < N; i++) {
			int direction = rand.nextInt(4);
			///for every step we generate a new random direction
			
			switch (direction) {///each case represents a counter of a direction that can be increased and decreased.
			case 0:
				y++;
				break;
			case 1:
				y--;
				break;
			case 2:
				x++;
				break;
			case 3:
				x--;
				break;
		
			}
			
			System.out.println("(" + x + ", " + y + ")");///print the coords for each step
			
			squared_dist = x*x + y*y;///calculate the squared distance
		}
		
		System.out.println("Squared distance = " + squared_dist);
	}

}

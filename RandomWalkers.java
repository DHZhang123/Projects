import java.util.Random;

public class RandomWalkers {
///write a program that simulates T amount of drunk walks that last for N amount of steps each
	
	public static void main(String[] args) {
		int N = Integer.parseInt(args[0]);
		int T = Integer.parseInt(args[1]);
		double totalsquared_dist = 0;
		//initialize
		
		Random rand = new Random();
		///set random to rand
		
		for (int z = 0; z < T; z++) {
			///loop through T amount of walks
			int x = 0;
			int y = 0;
			int squared_dist = 0;
			
			for (int i = 0; i < N; i++) {
				int direction = rand.nextInt(4);
				///for every step we generate a new random direction
				
				switch (direction) {
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
			}	
		
			squared_dist = (x*x) + (y*y);
			totalsquared_dist += squared_dist;
			///find total squared_dist
		}
	
		double meansquared_dist = totalsquared_dist / T;
		///calculate mean
		
		System.out.println("Mean squared distance = " + meansquared_dist);
}
}

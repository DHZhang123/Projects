public class Checkerboard {
	public static void main(String[] args) {
	///write a program that outputs a checker-board of alternating * and spaces. 
		
		int N = Integer.parseInt(args[0]);
		///initialize int N
		for (int x = 0; x < N; x++) {
			for (int y = 0; y < N; y++) {
		///nested for loop
				if ((x + y) % 2 == 0) {
					System.out.print("*");
				} else {
					System.out.print(" * ");
				}/// alternate between * and " * "	
			}
			System.out.println(); ///print rows on separate lines	
		}
}
}
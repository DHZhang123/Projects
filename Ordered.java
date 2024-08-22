
public class Ordered {
	public static void main(String[] args) {
	///write a program to order 3 ints in decreasing or increasing order
		
	int x = Integer.parseInt(args[0]);
	int y = Integer.parseInt(args[1]);
	int z = Integer.parseInt(args[2]);
	///initialize ints x, y, z
	
	boolean isOrdered = true; ///initialize isOrdered as a boolean, true
	if (((x > y) && (y > z)) || ((x < y) && (y < z))) {
		System.out.println(isOrdered);
	}///if statement to determine of the 3 ints are ordered.
	
	else {
	System.out.println("false");	
	
	}
}
}
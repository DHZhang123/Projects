
public class RGBtoCMYK {
	public static void main(String[] args ) {
	///write a program to convert RGB values to CMYK
		
		double red = Integer.parseInt(args[0]);
		double green = Integer.parseInt(args[1]);
		double blue = Integer.parseInt(args[2]);
		double cyan;
		double magenta;
		double yellow;
		double black;
		double white = Math.max(Math.max(red/255.0, green/255.0), blue/255.0);
		///initialize our colors as doubles
		
		if (white == 0) {
			cyan = 0;
			magenta = 0;
			yellow = 0;
			black = 1;
		}///set the base cases for our colors if white = 0
		else {
			cyan  = (white - (red/255.0))/white;
			magenta = (white - (green/255.0))/white;
			yellow = (white - (blue/255.0))/white;
			black = 1 - white;
		}///calculate our color values
		
		System.out.println(cyan);
		System.out.println(magenta);
		System.out.println(yellow);
		System.out.println(black);
	}

}


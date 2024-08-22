public class Sierpinski {
	//Draw a sierpinski gasket of level n
	
	public static void sierpinski(double x0, double y0, double x1, double y1, double x2, double y2, int level) {
		//base case of level 0
		if (level == 0)
			triangle(x0, y0, x1, y1, x2, y2);
		else {
		//calculate midpoints for all the sides
			double mx0 = (x0 + x1) / 2;
			double my0 = (y0 + y1) / 2;
			double mx1 = (x1 + x2) / 2;
			double my1 = (y1 + y2) / 2;
			double mx2 = (x2 + x0) / 2;
			double my2 = (y2 + y0) / 2;

			// Recursively call sierpinski method for each smaller triangle
			sierpinski(x0, y0, mx0, my0, mx2, my2, level - 1);
			sierpinski(mx0, my0, x1, y1, mx1, my1, level - 1);
			sierpinski(mx2, my2, mx1, my1, x2, y2, level - 1);
		}

	}

	public static void triangle(double x0, double y0, double x1, double y1, double x2, double y2) {
		double[] x = { x0, x1, x2 };
		double[] y = { y0, y1, y2 };
		StdDraw.filledPolygon(x, y);
	}

	public static void main(String[] args) {
		int n = Integer.parseInt(args[0]);
		sierpinski(0, 0, 1, 0, .5, .866, n);
	}
}

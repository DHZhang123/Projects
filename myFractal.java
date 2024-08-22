public class myFractal {

    // Recursive method to draw a square(grid) fractal
    public static void drawSquare(double x, double y, double size, int level) {
        // Base case: if recursion reaches level 0, draw a square and return
        if (level == 0) {
            StdDraw.square(x + size / 2, y + size / 2, size / 2);
        } else {
            // Calculate size of smaller square
            double newSize = size / 2;
            
            // Recursively call drawSquare for each quadrant
            drawSquare(x, y, newSize, level - 1); // Top-left quadrant
            drawSquare(x + newSize, y, newSize, level - 1); // Top-right quadrant
            drawSquare(x, y + newSize, newSize, level - 1); // Bottom-left quadrant
            drawSquare(x + newSize, y + newSize, newSize, level - 1); // Bottom-right quadrant
        }
    }

    public static void main(String[] args) {
        int n = Integer.parseInt(args[0]); // Level of recursion
        double size = 1; // Size of the initial square

       

        // Call the drawSquare method to draw the square fractal
        drawSquare(0, 0, size, n);
    }
}
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.math.*;


public class NBody {

	/**
	 * @param args
	 * @throws FileNotFoundException 
	 */
	
	public static void main(String[] args) throws FileNotFoundException {

		double big_t = Double.parseDouble(args[0]);
		double delta_t = Double.parseDouble(args[1]);

		String resourceFolder = "resources/";
		String fileName = resourceFolder + args[2];
		FileInputStream fileInputStream = new FileInputStream(fileName);
		System.setIn(fileInputStream);

		// Use StdIn to read from the file.
		int numBodies = StdIn.readInt();
		double universeRadius = StdIn.readDouble();

		// planetXpos  = array of double for X positions
		double[] PosX = new double[numBodies];
		// planetYpos  = array of double for Y positions
		double[] PosY = new double[numBodies];
		// planetXvel  = array of double for X velocities
		double[] VelX = new double[numBodies];
		// planetYvel  = array of double for Y velocities
		double[] VelY = new double[numBodies];
		// planetMass  = array of double for each planets mass
		double[] Mass = new double[numBodies];
		// planetNames = array of strings for each planets name
		String[] Pict = new String[numBodies];

		// Read in each planet
		for (int i=0; i<numBodies; i++) {
			PosX[i] = StdIn.readDouble();
			PosY[i] = StdIn.readDouble();
			VelX[i] = StdIn.readDouble();
			VelY[i] = StdIn.readDouble();
			Mass[i] = StdIn.readDouble();
			Pict[i] = StdIn.readString();
		}
		
		StdDraw.setXscale(-universeRadius, universeRadius);
		StdDraw.setYscale(-universeRadius, universeRadius);
		StdDraw.picture(0, 0, "resources/starfield.jpg");
	
		StdDraw.enableDoubleBuffering();
		StdAudio.playInBackground("resources/2001.wav");
	
		double G = 6.67 * Math.pow(10,  -11);
	
		
		
		//declare forceX array for all bodies
		double[] Xforces = new double[numBodies];
		//declare forceY array for all bodies
		double[] Yforces = new double[numBodies];
		
		
		double time_step = 0;
		
		while (time_step < big_t) {

		// One ITERATION
		for (int i = 0; i < numBodies; i++) {// start if1

			Xforces[i] = 0;
			Yforces[i] = 0;

			for (int j = 0; j < numBodies; j++) {// start if2
				if (i == j) continue;// start if1
				// calc deltaX
				double deltaX = PosX[j] - PosX[i];
				// calc deltaY
				double deltaY = PosY[j] - PosY[i];
				// calc R
				double R = Math.sqrt(deltaX * deltaX + deltaY * deltaY);
				// calc force
				double force = (G * Mass[i] * Mass[j]) / (R * R);		
				double forceX = force * (deltaX / R);
				// calc forceX
				Xforces[i] += forceX;
				// calc forceY
				double forceY = force * (deltaY / R);
				Yforces[i] += forceY;

				} // end if1
			} // end if2
			

		for (int i=0; i<numBodies; i++) {
			//single body operation
			//calculate accelerationX for each body
			double accelX = Xforces[i]/Mass[i];
			//calculate accelerationY for each body
			double accelY = Yforces[i]/Mass[i];
			//calculate new velX
			VelX[i] = VelX[i] + (delta_t * accelX);
			//calculate new velY
			VelY[i] = VelY[i] + (delta_t * accelY);
			//calculate new positionX...update pos
			PosX[i] = PosX[i] + (delta_t * VelX[i]);
			//calculate new positionY...update pos
			PosY[i] = PosY[i] + (delta_t * VelY[i]);
			
		}// end for 2
		//we now have new positions for X and Y
		
		StdDraw.picture(0,  0, "resources/starfield.jpg");
		for (int i=0; i<numBodies; i++) {
			//for each body, given posX/posY, draw that body			
			StdDraw.picture(PosX[i], PosY[i], "resources/" + Pict[i]);
		
			
		}
		
		StdDraw.show();///StdDraw.show() won't take an integer argument even though I used the .jar file provided.
		time_step = time_step + delta_t;
		
		}//end while
		
		// Print final state of the universe
		for (int i = 0; i < numBodies; i++) {
			System.out.println("-----");
			System.out.println(Pict[i]);
			System.out.println("X-Pos: " + PosX[i]);
			System.out.println("Y-Pos: " + PosY[i]);
			System.out.println("X-Vel: " + VelX[i]);
			System.out.println("Y-Vel: " + VelY[i]);
		}

	}
}
///StdDraw.show(30) doesn't work for some reason. I used the .jar file provided for this project. '
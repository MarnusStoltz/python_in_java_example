package test;

import org.apache.log4j.Logger;
import org.apache.log4j.BasicConfigurator;
import org.python.util.PythonInterpreter;
import org.python.core.*;
import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;

public class HelloWorld {

    static Logger logger = Logger.getLogger(HelloWorld.class);
    public static void main(String[] args) {

            String pythonScriptPath = "src/test.py";
            String network = "(((A[&Theta=0.010616363254683646]:0.03492794012453921)#H1[&Theta=0.011848389331849283,gamma=0.3756543450852222]:0.18666241422072055,(#H1[&Theta=0.006253316339164445]:0.0010641504386878914,(B[&Theta=0.007822837720358048]:0.02365017569992867,C[&Theta=0.0027920310539736154]:0.02365017569992867)S2[&Theta=0.014941600389555195]:0.012341914863298432)S3[&Theta=0.0036267284387002346]:0.18559826378203265)S1[&Theta=0.013301263668811238]:2.7784096456547402);";
            String rho = "#H1 0.3756543450852222;";
            String data = "C 2 2;A 1 2;B 0 2;";
            float likelihood;

            try {
                ProcessBuilder processBuilder = new ProcessBuilder("python3", pythonScriptPath, network, rho, data);
                Process process = processBuilder.start();
    
                // Read the output from the Python script
                InputStream inputStream = process.getInputStream();
                BufferedReader reader = new BufferedReader(new InputStreamReader(inputStream));
                String line;
                while ((line = reader.readLine()) != null) {
                    likelihood = Float.parseFloat(line);
                }
    
                // Wait for the Python process to complete
                int exitCode = process.waitFor();
                //System.out.println("Python script execution completed with exit code: " + exitCode);
            } catch (IOException | InterruptedException e) {
                e.printStackTrace();
            }
        }
}
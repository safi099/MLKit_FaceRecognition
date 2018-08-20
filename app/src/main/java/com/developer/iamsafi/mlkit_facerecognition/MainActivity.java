package com.developer.iamsafi.mlkit_facerecognition;

import android.app.ProgressDialog;
import android.content.Intent;
import android.graphics.Bitmap;
import android.os.AsyncTask;
import android.provider.MediaStore;
import android.support.annotation.NonNull;
import android.support.annotation.Nullable;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import com.developer.iamsafi.mlkit_facerecognition.helper.DataSet;
import com.developer.iamsafi.mlkit_facerecognition.model.Face_Features;
import com.developer.iamsafi.mlkit_facerecognition.neural_network.MultiLayerPerceptron;
import com.developer.iamsafi.mlkit_facerecognition.neural_network.transfer_functions.SigmoidalTransfer;
import com.google.android.gms.tasks.OnFailureListener;
import com.google.android.gms.tasks.OnSuccessListener;
import com.google.android.gms.tasks.Task;
import com.google.android.gms.vision.face.Landmark;
import com.google.firebase.FirebaseApp;
import com.google.firebase.ml.vision.FirebaseVision;
import com.google.firebase.ml.vision.common.FirebaseVisionImage;
import com.google.firebase.ml.vision.common.FirebaseVisionPoint;
import com.google.firebase.ml.vision.face.FirebaseVisionFace;
import com.google.firebase.ml.vision.face.FirebaseVisionFaceDetector;
import com.google.firebase.ml.vision.face.FirebaseVisionFaceDetectorOptions;
import com.google.firebase.ml.vision.face.FirebaseVisionFaceLandmark;

import java.util.ArrayList;
import java.util.List;

public class MainActivity extends AppCompatActivity {
    TextView tv_counter;
    Button btn_take;
    ImageView img_photo;
    int pic_counter = 0;
    private int REQUEST_IMAGE_CAPTURE = 10;
    private FirebaseVisionImage image;
    private FirebaseVisionFaceDetector detector;
    ArrayList<Face_Features> features;
    DataSet dataSet = new DataSet();

    float[][] input = new float[5][21];
    Button btn_train;
    private MultiLayerPerceptron neuralNetwork;
    private ProgressDialog progressDialog;
    private int MAX_ITERATIONS = 20000;
    private double error;
    private Button btn_recognize;
    private int REQUEST_RECOGNIZER = 20;
    private double[] inputrecognizer = new double[21];

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        bindView();
        FirebaseApp.initializeApp(getApplicationContext());
        FirebaseVisionFaceDetectorOptions options =
                new FirebaseVisionFaceDetectorOptions.Builder()
                        .setModeType(FirebaseVisionFaceDetectorOptions.ACCURATE_MODE)
                        .setLandmarkType(FirebaseVisionFaceDetectorOptions.ALL_LANDMARKS)
                        .setClassificationType(FirebaseVisionFaceDetectorOptions.ALL_CLASSIFICATIONS)
                        .setMinFaceSize(0.15f)
                        .setTrackingEnabled(true)
                        .build();

        detector = FirebaseVision.getInstance().getVisionFaceDetector(options);

        btn_take.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                // Toast.makeText(getApplicationContext(), "You have to take 5 pictures to train data", Toast.LENGTH_LONG).show();
                if (pic_counter == 5) {
                    btn_train.setEnabled(true);
                    btn_take.setEnabled(false);
                    displayFeatures();
                } else {
                    Intent takePictureIntent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
                    if (takePictureIntent.resolveActivity(getPackageManager()) != null) {
                        startActivityForResult(takePictureIntent, REQUEST_IMAGE_CAPTURE);
                    }
                }
            }//onclick

        });
        //=========================================
        btn_train.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                int[] layers = new int[]{21, 2, 1};
                neuralNetwork = new MultiLayerPerceptron(layers, 0.4, new SigmoidalTransfer());
                progressDialog = ProgressDialog.show(MainActivity.this, "Training", "Please wait...");
                new TrainTask().execute();
            }
        });
        //========================================
        btn_recognize.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                Intent takePictureIntent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
                if (takePictureIntent.resolveActivity(getPackageManager()) != null) {
                    startActivityForResult(takePictureIntent, REQUEST_RECOGNIZER);
                }
            }
        });
    }//oncreate()

    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        if (requestCode == REQUEST_IMAGE_CAPTURE && resultCode == RESULT_OK) {
            Bundle extras = data.getExtras();
            Bitmap imageBitmap = (Bitmap) extras.get("data");
            img_photo.setImageBitmap(imageBitmap);
            //img_photo.setImageBitmap(ImageProcessor.process(imageBitmap));
            image = FirebaseVisionImage.fromBitmap(imageBitmap);

            //Detected the Images
            Task<List<FirebaseVisionFace>> result =
                    detector.detectInImage(image)
                            .addOnSuccessListener(
                                    new OnSuccessListener<List<FirebaseVisionFace>>() {
                                        @Override
                                        public void onSuccess(List<FirebaseVisionFace> faces) {
                                            // Task completed successfully
                                            try {
                                                for (FirebaseVisionFace face : faces) {
                                                    Log.i("check", "Gathering Features");
                                                    Face_Features face_features = new Face_Features();

                                                    FirebaseVisionFaceLandmark firebaseVisionCloudLandmark = face.getLandmark(Landmark.LEFT_EYE);
                                                    face_features.setPoint_left_eyes(firebaseVisionCloudLandmark.getPosition());
                                                    firebaseVisionCloudLandmark = face.getLandmark(Landmark.RIGHT_EYE);
                                                    face_features.setPoint_right_eyes(firebaseVisionCloudLandmark.getPosition());
                                                    firebaseVisionCloudLandmark = face.getLandmark(Landmark.LEFT_CHEEK);
                                                    face_features.setPoint_left_cheek(firebaseVisionCloudLandmark.getPosition());
                                                    firebaseVisionCloudLandmark = face.getLandmark(Landmark.RIGHT_CHEEK);
                                                    face_features.setPoint_right_cheek(firebaseVisionCloudLandmark.getPosition());
                                                    firebaseVisionCloudLandmark = face.getLandmark(Landmark.NOSE_BASE);
                                                    face_features.setPoint_nose_base(firebaseVisionCloudLandmark.getPosition());
                                                    firebaseVisionCloudLandmark = face.getLandmark(Landmark.LEFT_MOUTH);
                                                    face_features.setPoint_left_mouth(firebaseVisionCloudLandmark.getPosition());
                                                    firebaseVisionCloudLandmark = face.getLandmark(Landmark.RIGHT_MOUTH);
                                                    face_features.setPoint_right_mouth(firebaseVisionCloudLandmark.getPosition());
                                                    features.add(face_features);
                                                    Log.i("check", "Picture Counter = " + pic_counter);
                                                    Log.i("check", "Picture Counter = " + pic_counter);
                                                    if (pic_counter != 5) {
                                                        Toast.makeText(MainActivity.this, " " + (pic_counter + 1) + "/5 pictures to train dataset", Toast.LENGTH_SHORT).show();
                                                        calculateFeaturesDistance();
                                                    } else {
                                                        Toast.makeText(MainActivity.this, "Training Completed", Toast.LENGTH_SHORT).show();
                                                        //=====Calculating Distance from Face Features
                                                        displayFeatures();

                                                    }
                                                    pic_counter++;

                                                    Log.i("check", "Features of Person = " + features.size());

                                                }//for faces loop
                                            } catch (NullPointerException ex) {
                                                Toast.makeText(getApplicationContext(), "Please take the stable image try again", Toast.LENGTH_SHORT).show();
                                            }
                                        }
                                    })
                            .addOnFailureListener(
                                    new OnFailureListener() {
                                        @Override
                                        public void onFailure(@NonNull Exception e) {
                                            // Task failed with an exception
                                            Toast.makeText(MainActivity.this, "Face is not detected", Toast.LENGTH_LONG).show();

                                        }
                                    });
        } else if (requestCode == REQUEST_RECOGNIZER && resultCode == RESULT_OK) {
            Bundle extras = data.getExtras();
            Bitmap imageBitmap = (Bitmap) extras.get("data");
            img_photo.setImageBitmap(imageBitmap);
            //img_photo.setImageBitmap(ImageProcessor.process(imageBitmap));
            image = FirebaseVisionImage.fromBitmap(imageBitmap);

            //Detected the Images
            Task<List<FirebaseVisionFace>> result =
                    detector.detectInImage(image)
                            .addOnSuccessListener(
                                    new OnSuccessListener<List<FirebaseVisionFace>>() {
                                        @Override
                                        public void onSuccess(List<FirebaseVisionFace> faces) {
                                            // Task completed successfully
                                            try {
                                                for (FirebaseVisionFace face : faces) {
                                                    Log.i("check", "Gathering Features");
                                                    Face_Features face_features = new Face_Features();

                                                    FirebaseVisionFaceLandmark firebaseVisionCloudLandmark = face.getLandmark(Landmark.LEFT_EYE);
                                                    face_features.setPoint_left_eyes(firebaseVisionCloudLandmark.getPosition());
                                                    firebaseVisionCloudLandmark = face.getLandmark(Landmark.RIGHT_EYE);
                                                    face_features.setPoint_right_eyes(firebaseVisionCloudLandmark.getPosition());
                                                    firebaseVisionCloudLandmark = face.getLandmark(Landmark.LEFT_CHEEK);
                                                    face_features.setPoint_left_cheek(firebaseVisionCloudLandmark.getPosition());
                                                    firebaseVisionCloudLandmark = face.getLandmark(Landmark.RIGHT_CHEEK);
                                                    face_features.setPoint_right_cheek(firebaseVisionCloudLandmark.getPosition());
                                                    firebaseVisionCloudLandmark = face.getLandmark(Landmark.NOSE_BASE);
                                                    face_features.setPoint_nose_base(firebaseVisionCloudLandmark.getPosition());
                                                    firebaseVisionCloudLandmark = face.getLandmark(Landmark.LEFT_MOUTH);
                                                    face_features.setPoint_left_mouth(firebaseVisionCloudLandmark.getPosition());
                                                    firebaseVisionCloudLandmark = face.getLandmark(Landmark.RIGHT_MOUTH);
                                                    face_features.setPoint_right_mouth(firebaseVisionCloudLandmark.getPosition());
                                                    features.add(face_features);
                                                    calculateFeaturesDistance();
                                                    progressDialog = ProgressDialog.show(MainActivity.this, "Recognizing", "Please wait...");
                                                    new RecognizeTask().execute();

                                                }//for faces loop
                                            } catch (NullPointerException ex) {
                                                Toast.makeText(getApplicationContext(), "Please take the stable image try again", Toast.LENGTH_SHORT).show();
                                            }
                                        }
                                    })
                            .addOnFailureListener(
                                    new OnFailureListener() {
                                        @Override
                                        public void onFailure(@NonNull Exception e) {
                                            // Task failed with an exception
                                            Toast.makeText(MainActivity.this, "Face is not detected", Toast.LENGTH_LONG).show();

                                        }
                                    });
        }
        super.onActivityResult(requestCode, resultCode, data);
    }

    private void displayFeatures() {
        for (int i = 0; i < 5; i++) {
            System.out.print("Image#" + (i + 1) + ": ");
            for (int j = 0; j < 21; j++) {
                System.out.print(+input[i][j] + "  ");
            }
            System.out.println();
        }
    }

    private void calculateFeaturesDistance() {
        for (int i = 0; i < features.size(); i++) {
            Face_Features face_features = new Face_Features();
            String Leyes = getPoints(features.get(i).getPoint_left_eyes());
            String Reyes = getPoints(features.get(i).getPoint_right_eyes());
            String Lmouth = getPoints(features.get(i).getPoint_left_mouth());
            String Rmouth = getPoints(features.get(i).getPoint_right_mouth());
            String Lcheek = getPoints(features.get(i).getPoint_left_cheek());
            String Rcheek = getPoints(features.get(i).getPoint_right_cheek());
            String nosebase = getPoints(features.get(i).getPoint_nose_base());
            ArrayList<Float> dist = new ArrayList<>();

            float eyes_dist = FeatureDistance(Leyes, Reyes);
            dataSet.addValue(eyes_dist);
            dist.add(eyes_dist);
            float cheek_dist = FeatureDistance(Lcheek, Rcheek);
            dataSet.addValue(cheek_dist);
            dist.add(cheek_dist);
            float mouth_dist = FeatureDistance(Lmouth, Rmouth);
            dataSet.addValue(mouth_dist);
            dist.add(mouth_dist);
            float Leyes2nose_dist = FeatureDistance(Leyes, nosebase);
            dataSet.addValue(Leyes2nose_dist);
            dist.add(Leyes2nose_dist);
            float Reyes2nose_dist = FeatureDistance(Reyes, nosebase);
            dataSet.addValue(Reyes2nose_dist);
            dist.add(Reyes2nose_dist);
            float Leyes2Lcheek_dist = FeatureDistance(Leyes, Lcheek);
            dataSet.addValue(Leyes2Lcheek_dist);
            dist.add(Leyes2Lcheek_dist);
            float Reyes2Rcheek_dist = FeatureDistance(Reyes, Rcheek);
            dataSet.addValue(Reyes2Rcheek_dist);
            dist.add(Reyes2Rcheek_dist);
            float Leyes2Lmouth_dist = FeatureDistance(Leyes, Lmouth);
            dataSet.addValue(Leyes2Lmouth_dist);
            dist.add(Leyes2Lmouth_dist);
            float Reyes2Rmouth_dist = FeatureDistance(Reyes, Rmouth);
            dataSet.addValue(Reyes2Rmouth_dist);
            dist.add(Reyes2Rmouth_dist);
            float Lcheek2Lmouth_dist = FeatureDistance(Lcheek, Lmouth);
            dataSet.addValue(Lcheek2Lmouth_dist);
            dist.add(Lcheek2Lmouth_dist);
            float Rcheek2Rmouth_dist = FeatureDistance(Rcheek, Rmouth);
            dataSet.addValue(Rcheek2Rmouth_dist);
            dist.add(Rcheek2Rmouth_dist);
            float Lcheek2nose_dist = FeatureDistance(Lcheek, nosebase);
            dataSet.addValue(Lcheek2nose_dist);
            dist.add(Lcheek2nose_dist);
            float Rcheek2nose_dist = FeatureDistance(Rcheek, nosebase);
            dataSet.addValue(Rcheek2nose_dist);
            dist.add(Rcheek2nose_dist);
            float Lmouth2nose_dist = FeatureDistance(Lmouth, nosebase);
            dataSet.addValue(Lmouth2nose_dist);
            dist.add(Lmouth2nose_dist);
            float Rmouth2nose_dist = FeatureDistance(Rmouth, nosebase);
            dataSet.addValue(Rmouth2nose_dist);
            dist.add(Rmouth2nose_dist);
            String centermouth = centerPoint(Lmouth, Rmouth);
            float Rmouth2centermouth_dist = FeatureDistance(Rmouth, centermouth);
            dataSet.addValue(Rmouth2centermouth_dist);
            dist.add(Rmouth2centermouth_dist);
            float Lmouth2centermouth_dist = FeatureDistance(Lmouth, centermouth);
            dataSet.addValue(Lmouth2centermouth_dist);
            dist.add(Lmouth2centermouth_dist);
            float nose2centermouth = FeatureDistance(nosebase, centermouth);
            dataSet.addValue(nose2centermouth);
            dist.add(nose2centermouth);
            String centereyes = centerPoint(Leyes, Reyes);
            float Leyes2centereye_dist = FeatureDistance(Leyes, centereyes);
            dataSet.addValue(Leyes2centereye_dist);
            dist.add(Leyes2centereye_dist);
            float Reyes2centereye_dist = FeatureDistance(Reyes, centereyes);
            dataSet.addValue(Reyes2centereye_dist);
            dist.add(Reyes2centereye_dist);
            float nose2centereye_dist = FeatureDistance(nosebase, centereyes);
            dataSet.addValue(nose2centereye_dist);
            dist.add(nose2centereye_dist);
            normalizeDataSet(dataSet, dist);

        }
    }

    private void normalizeDataSet(DataSet dataSet, ArrayList<Float> dist) {

        for (int i = 0; i < dist.size(); i++) {
            inputrecognizer[i] = ((dist.get(i) - dataSet.getSmallest()) / (dataSet.getLargest() - dataSet.getSmallest()));
            if (pic_counter != 5)
                input[pic_counter][i] = ((dist.get(i) - dataSet.getSmallest()) / (dataSet.getLargest() - dataSet.getSmallest()));
        }


    }

    private String centerPoint(String lvalue, String rvalue) {
        String[] parts1 = lvalue.split(",");
        String[] parts2 = rvalue.split(",");
        return ((Float.parseFloat(parts2[0]) + Float.parseFloat(parts1[0])) / 2) + "," + ((Float.parseFloat(parts2[1]) + Float.parseFloat(parts1[1])) / 2);

    }

    private float FeatureDistance(String leyes, String reyes) {
        String[] parts1 = leyes.split(",");
        String[] parts2 = reyes.split(",");
        float dist = (float) Math.sqrt(
                Math.pow(Float.parseFloat(parts2[0]) - Float.parseFloat(parts1[0]), 2) +
                        Math.pow(Float.parseFloat(parts2[1]) - Float.parseFloat(parts1[1]), 2));
        return dist;
    }

    @NonNull
    private String getPoints(FirebaseVisionPoint point_feature) {
        return point_feature.getX() + "," + point_feature.getY();
    }

    private void bindView() {
        tv_counter = findViewById(R.id.counter);
        btn_take = findViewById(R.id.take_photo);
        img_photo = findViewById(R.id.img_camera);
        btn_train = findViewById(R.id.train);
        btn_recognize = findViewById(R.id.recognize);
        features = new ArrayList<>();

    }


    private class TrainTask extends AsyncTask<Void, Void, Double> {

        protected void onPostExecute(Double result) {
            Log.i("check", "Error: " + result);
            progressDialog.hide();
            progressDialog.cancel();
            btn_train.setEnabled(false);
            btn_recognize.setEnabled(true);
        }

        @Override
        protected Double doInBackground(Void... voids) {
            double[] inputs = new double[21];
            for (int i = 0; i < MAX_ITERATIONS; i++) {
                for (int j = 0; j < 5; j++) {
                    for (int k = 0; k < 21; k++) {
                        inputs[k] = input[j][k];
                    }
                }
                if (inputs == null) {
                    System.out.println("Cant find images features");
                    continue;
                }
                double[] output = new double[1];
                // Training
                error = neuralNetwork.backPropagate(inputs, output);
                System.out.println("Error is " + error);

            }
            System.out.println("Learning completed!");
            return error;
        }
    }

    private class RecognizeTask extends AsyncTask<Void, Void, Double> {
        @Override
        protected void onPostExecute(Double aDouble) {
            super.onPostExecute(aDouble);
            progressDialog.hide();
            progressDialog.cancel();
            Log.i("check", "Correction: " + aDouble);
            if (aDouble > 0.006)
                Toast.makeText(getApplicationContext(), "You are not the person on which I was trained..", Toast.LENGTH_SHORT).show();
            else
                Toast.makeText(getApplicationContext(), "I Recognize that picture", Toast.LENGTH_SHORT).show();

        }

        @Override
        protected Double doInBackground(Void... voids) {
            int correct = 0;


//            double[] inputs = ImageProcessingBW.loadImage("/home/dak/workspace/MultiLayersPerceptronLib/img/test.png", size, size);
            double[] output = neuralNetwork.execute(inputrecognizer);

            int max = 0;
            for (int i = 0; i < 1; i++)
                if (output[i] > output[max]) {
                    max = i;
                }

            System.out.println("Il valore massimo e' " + output[max] + " pattern " + (max + 1));
//            return (double) ((double) (1*21) - (double) correct) / (double) (1*21);
            return output[max];
        }
    }
}

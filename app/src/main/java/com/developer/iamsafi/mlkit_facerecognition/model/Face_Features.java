package com.developer.iamsafi.mlkit_facerecognition.model;

import com.google.firebase.ml.vision.common.FirebaseVisionPoint;

public class Face_Features {
    FirebaseVisionPoint point_left_eyes;
    FirebaseVisionPoint point_right_eyes;
    FirebaseVisionPoint point_left_cheek;
    FirebaseVisionPoint point_right_cheek;
    FirebaseVisionPoint point_left_mouth;
    FirebaseVisionPoint point_right_mouth;
    FirebaseVisionPoint point_nose_base;

    public FirebaseVisionPoint getPoint_left_eyes() {
        return point_left_eyes;
    }

    public void setPoint_left_eyes(FirebaseVisionPoint point_left_eyes) {
        this.point_left_eyes = point_left_eyes;
    }

    public FirebaseVisionPoint getPoint_right_eyes() {
        return point_right_eyes;
    }

    public void setPoint_right_eyes(FirebaseVisionPoint point_right_eyes) {
        this.point_right_eyes = point_right_eyes;
    }

    public FirebaseVisionPoint getPoint_left_cheek() {
        return point_left_cheek;
    }

    public void setPoint_left_cheek(FirebaseVisionPoint point_left_cheek) {
        this.point_left_cheek = point_left_cheek;
    }

    public FirebaseVisionPoint getPoint_right_cheek() {
        return point_right_cheek;
    }

    public void setPoint_right_cheek(FirebaseVisionPoint point_right_cheek) {
        this.point_right_cheek = point_right_cheek;
    }

    public FirebaseVisionPoint getPoint_left_mouth() {
        return point_left_mouth;
    }

    public void setPoint_left_mouth(FirebaseVisionPoint point_left_mouth) {
        this.point_left_mouth = point_left_mouth;
    }

    public FirebaseVisionPoint getPoint_right_mouth() {
        return point_right_mouth;
    }

    public void setPoint_right_mouth(FirebaseVisionPoint point_right_mouth) {
        this.point_right_mouth = point_right_mouth;
    }

    public FirebaseVisionPoint getPoint_nose_base() {
        return point_nose_base;
    }

    public void setPoint_nose_base(FirebaseVisionPoint point_nose_base) {
        this.point_nose_base = point_nose_base;
    }
}

package com.developer.iamsafi.mlkit_facerecognition.helper;

public class DataSet {
    private float smallest = Float.MAX_VALUE;
    private float largest = Integer.MIN_VALUE;

    /**
     * Adds in integer to sequence.
     *
     * @param x the integer added
     */
    public void addValue(float x) {
        smallest = Math.min(smallest, x);
        largest = Math.max(largest, x);
    }

    /**
     * Returns the smallest value.
     *
     * @return the smallest value
     */
    public float getSmallest() {
        return smallest;
    }

    /**
     * Returns the largest value.
     *
     * @return the largest value
     */
    public float getLargest() {
        return largest;
    }
}

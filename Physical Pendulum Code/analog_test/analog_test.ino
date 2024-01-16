void setup() {
  //analogReadResolution(12); //change to 12-bit resolution
  Serial.begin(9600);
  
}

void loop() {
  // put your main code here, to run repeatedly:
  int reading = analogRead(A2); // returns a value between 0-4095
  Serial.println(reading);
}


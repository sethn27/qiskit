int directionPin = 12;
int pwmPin = 3;
int brakePin = 9;

//uncomment if using channel B, and remove above definitions
//int directionPin = 13;
//int pwmPin = 11;
//int brakePin = 8;

//boolean to switch direction
bool directionState;
int encoderA = 18;
int encoderB = 19;
int pulses;
int pulsesChanged = 0;

#define total 1040 //13x20x4

void setup() {
Serial.begin(115200);
pinMode(encoderA, INPUT_PULLUP);
pinMode(encoderB, INPUT_PULLUP);
//define pins
pinMode(directionPin, OUTPUT);
pinMode(pwmPin, OUTPUT);
pinMode(brakePin, OUTPUT);

attachInterrupt(digitalPinToInterrupt(18), A_CHANGE, CHANGE);
attachInterrupt(digitalPinToInterrupt(19), B_CHANGE, CHANGE);
}

void loop() {

//change direction every loop()
directionState = !directionState;

//write a low state to the direction pin (13)
if(directionState == false){
  digitalWrite(directionPin, LOW);
}

//write a high state to the direction pin (13)
else{
  digitalWrite(directionPin, HIGH);
}

//release breaks
digitalWrite(brakePin, LOW);

//set work duty for the motor
analogWrite(pwmPin, 0);
int a = digitalRead(encoderA);
int b = digitalRead(encoderB);
Serial.println("start");
Serial.println(a);
Serial.println(b);
delay(50);
Serial.println(pulses);

//activate breaks
digitalWrite(brakePin, HIGH);

//set work duty for the motor to 0 (off)
analogWrite(pwmPin, 0);

delay(2000);
}

void A_CHANGE(){
  if( digitalRead(encoderB) == 0 ) {
    if ( digitalRead(encoderA) == 0 ) {
      // A fell, B is low
      pulses--; // moving reverse
    } else {
      // A rose, B is low
      pulses++; // moving forward
    }
  }else {
    if ( digitalRead(encoderA) == 0 ) {
      // B fell, A is high
      pulses++; // moving reverse
    } else {
      // B rose, A is high
      pulses--; // moving forward
    }
  }
  pulsesChanged = 1;
}

void B_CHANGE(){
  if ( digitalRead(encoderA) == 0 ) {
    if ( digitalRead(encoderB) == 0 ) {
      // B fell, A is low
      pulses++; // moving forward
    } else {
      // B rose, A is low
      pulses--; // moving reverse
    }
 } else {
    if ( digitalRead(encoderB) == 0 ) {
      // B fell, A is high
      pulses--; // moving reverse
    } else {
      // B rose, A is high
      pulses++; // moving forward
    }
  }
  pulsesChanged = 1;
}
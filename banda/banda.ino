int ledPin = 11;      // LED connected to digital pin 9
int DIR = 21;
int analogPin = 0;   // potentiometer connected to analog pin 3

int val = 0;         // variable to store the read value



void setup(){
  Serial.begin(9600);
  pinMode(ledPin, OUTPUT);   // sets the pin as output
  pinMode(DIR, OUTPUT);   // sets the pin as output
}



void loop() {
  if(Serial.available()){
    Serial.read();
    while(true){
      if(Serial.available()){
        Serial.read();
        break;
      }
      if (digitalRead(20) == HIGH) {
        digitalWrite(DIR, HIGH);
      }else{
        digitalWrite(DIR, LOW);
      }
    
      val = analogRead(analogPin);   // read the input pin
      int val1 = map(val, 0, 1023, 1500, 10);
    
      if (val1< 1500&&val>15) {
        digitalWrite(ledPin, HIGH); // analogRead values go from 0 to 1023, analogWrite values from 0 to 255
        delayMicroseconds(val1);
        digitalWrite(ledPin, LOW);
        delayMicroseconds(val1);
      }else{
        digitalWrite(ledPin, LOW);
      }  
    }
      
  }
  
  
}

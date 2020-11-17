#include <opencv2/opencv.hpp>
#include <raspicam_cv.h>
#include <iostream>
#include <chrono>
#include <ctime>
#include <wiringPi.h>
#include <softPwm.h>


using namespace std;
using namespace cv;
using namespace raspicam;

#define  MotorPin1 23
#define  MotorPin2 24
#define  MotorEnableDC1 25

#define MotorPin3 16
#define MotorPin4 20
#define MotorEnableDC2 21

// Image Processing variables
Mat frame, Matrix, framePers, frameGray, frameThresh, frameEdge, frameFinal, frameFinalDuplicate, frameFinalDuplicate1;
Mat ROILane, ROILaneEnd;
int LeftLanePos, RightLanePos, frameCenter, laneCenter, Result, laneEnd;

RaspiCam_Cv Camera;
stringstream ss;

vector<int> histogramLane;

Point2f Source[]={Point2f(125,180),Point2f(280,180),Point2f(115,220),Point2f(290,220)};
Point2f Destination[]={Point2f(120,0),Point2f(290,0),Point2f(120,240),Point2f(290,240)};

//Machine Learning variables
CascadeClassifier Stop_Cascade;
Mat frame_Stop, RoI_Stop, gray_Stop;
vector<Rect> Stop;
int dist_Stop;

void Setup ( int argc,char **argv, RaspiCam_Cv &Camera )
  {
    Camera.set ( CAP_PROP_FRAME_WIDTH,  ( "-w",argc,argv,400) );
    Camera.set ( CAP_PROP_FRAME_HEIGHT,  ( "-h",argc,argv,240 ) );
    Camera.set ( CAP_PROP_BRIGHTNESS, ( "-br",argc,argv,50 ) );
    Camera.set ( CAP_PROP_CONTRAST ,( "-co",argc,argv,50 ) );
    Camera.set ( CAP_PROP_SATURATION,  ( "-sa",argc,argv,50 ) );
    Camera.set ( CAP_PROP_GAIN,  ( "-g",argc,argv ,50 ) );
    Camera.set ( CAP_PROP_FPS,  ( "-fps",argc,argv,0));

}
void Perspective(){
	line(frame,Source[0],Source[1],Scalar(255,255,255),2);
	line(frame,Source[1],Source[3],Scalar(255,255,255),2);
	line(frame,Source[3],Source[2],Scalar(255,255,255),2);
	line(frame,Source[2],Source[0],Scalar(255,255,255),2);
	
	//line(frame,Destination[0],Destination[1],Scalar(0,255,0),2);
	//line(frame,Destination[1],Destination[3],Scalar(0,255,0),2);
	//line(frame,Destination[3],Destination[2],Scalar(0,255,0),2);
	//line(frame,Destination[2],Destination[0],Scalar(0,255,0),2);
	
	Matrix = getPerspectiveTransform(Source, Destination);
	warpPerspective(frame,framePers, Matrix, Size(350,240));
}
void Treshhold(){
	cvtColor(framePers,frameGray, COLOR_RGB2GRAY);
	inRange(frameGray,0,100,frameThresh);
	Canny(frameGray,frameEdge,0,300,3,false);
	add(frameThresh,frameEdge,frameFinal);
	cvtColor(frameFinal,frameFinal,COLOR_GRAY2RGB);
	
	}
void Histogram()
	{
	  histogramLane.resize(400);
	  histogramLane.clear();
	  
	  for(int i=0;i<frame.size().width;i+4){
		  ROILane=frameFinal(Rect(i,140,4,100));                                                                                              
		  divide(255,ROILane,ROILane);
		  histogramLane.push_back((int)(sum(ROILane)[0]));
		  
		  }
	
	}
void LaneFinder()
	{
	vector<int>:: iterator LeftPtr;
	LeftPtr= max_element(histogramLane.begin(),histogramLane.begin()+150);
	LeftLanePos=distance(histogramLane.begin(),LeftPtr);
	
	vector<int>:: iterator RightPtr;
	RightPtr= max_element(histogramLane.begin()+250,histogramLane.end());
	RightLanePos=distance(histogramLane.begin(),RightPtr);
	
	line(frameFinal,Point2f(LeftLanePos,0),Point2f(LeftLanePos,240),Scalar(0,255,0),2);
	line(frameFinal,Point2f(RightLanePos,0),Point2f(RightLanePos,240),Scalar(0,255,0),2);
	}
	
	

void Capture()
{

    Camera.grab();
    Camera.retrieve( frame);
    cvtColor(frame, frame_Stop, COLOR_BGR2RGB);
    cvtColor(frame, frame, COLOR_BGR2RGB);
	
}


void Stop_detection()
{
    if(!Stop_Cascade.load("/home/pi/Desktop/MACHINE LEARNING/Stop_cascade.xml"))
    {
	printf("Unable to open stop cascade file");
    }
    
    RoI_Stop = frame_Stop(Rect(100,0,200,160));
    cvtColor(RoI_Stop, gray_Stop, COLOR_RGB2GRAY);
    equalizeHist(gray_Stop, gray_Stop);
    Stop_Cascade.detectMultiScale(gray_Stop, Stop);
    
    for(int i=0; i<Stop.size(); i++)
    {
	Point P1(Stop[i].x, Stop[i].y);
	Point P2(Stop[i].x + Stop[i].width, Stop[i].y + Stop[i].height);
	
	rectangle(RoI_Stop, P1, P2, Scalar(0, 0, 255), 2);
	putText(RoI_Stop, "Stop Obj", P1, FONT_HERSHEY_PLAIN, 1,  Scalar(0, 0, 255, 255), 2);
	
	dist_Stop = (-0.923)*(P2.x-P1.x)+133.07;
	
       ss.str(" ");
       ss.clear();
       ss<<"D = "<<dist_Stop<<"cm";
       putText(RoI_Stop, ss.str(), Point2f(1,130), 0,1, Scalar(0,0,255), 2);
	
    }
    
}

int main(int argc,char **argv)
{
	if(wiringPiSetupGpio()==-1)
{
    cout<<"Setup wiring pi failed";
    return 1;
}
	Setup(argc,argv,Camera);
	
	pinMode(MotorPin1,PWM_OUTPUT);
	pinMode(MotorPin2,PWM_OUTPUT);
	pinMode(MotorEnableDC1,PWM_OUTPUT);

	pinMode(MotorPin3,PWM_OUTPUT);
	pinMode(MotorPin4,PWM_OUTPUT);
	pinMode(MotorEnableDC2,PWM_OUTPUT);

	softPwmCreate(MotorEnableDC1,0,1000);
	softPwmCreate(MotorEnableDC2,0,1000);
	
	
	softPwmWrite(MotorEnableDC1,1000);
    digitalWrite(MotorPin1,0);
    digitalWrite(MotorPin2,1);
   
    softPwmWrite(MotorEnableDC2,1000);
    digitalWrite(MotorPin3,0);
    digitalWrite(MotorPin4,1);
	cout<<"Connecting to Camera"<<endl;
	
	if(!Camera.open())
	{
		cout<<"Failed to connect"<<endl;
		return -1;
	}
	cout<<"Camera ID="<<Camera.getId()<<endl;
	while(1)
	{ 
	  		
	  auto start = std::chrono::system_clock::now();
      Capture();    
      Perspective();
      Treshhold();   
      Stop_detection();
  
      if(dist_Stop>40 && dist_Stop<50)
      {
			softPwmWrite(MotorEnableDC1,1000);
			digitalWrite(MotorPin1,0);
			digitalWrite(MotorPin2,0);
		   
			softPwmWrite(MotorEnableDC2,1000);
			digitalWrite(MotorPin3,0);
			digitalWrite(MotorPin4,0);
			goto Stop_Sign;
			cout<<"STOP" <<endl;
	  } 
	  Stop_Sign:
	  
	  namedWindow("orginal", WINDOW_KEEPRATIO);
      moveWindow("orginal", 0, 100);
      resizeWindow("orginal", 640,480);
      imshow("orginal",frame);
      
      namedWindow("Perspective", WINDOW_KEEPRATIO);
      moveWindow("Perspective", 640, 100);
      resizeWindow("Perspective", 640, 480);
      imshow("Perspective",framePers);
      
      namedWindow("FINAL", WINDOW_KEEPRATIO);
      moveWindow("FINAL", 1280, 100);
      resizeWindow("FINAL", 640, 480);
      imshow("FINAL",frameFinal);
      
      namedWindow("Stop", WINDOW_KEEPRATIO);
	  moveWindow("Stop", 1280, 580);
      resizeWindow("Stop", 640, 480);
      imshow("Stop", RoI_Stop);
	  
	  waitKey(1);
      auto end = std::chrono::system_clock::now();
	  std::chrono::duration<double> elapsed_seconds = end-start;
	  float t = elapsed_seconds.count();
	  int FPS = 4/t;
	  cout<<"FPS =" <<FPS<<endl;
      
      
      
	  
	  
	  }
	return 0;
	
	
	}
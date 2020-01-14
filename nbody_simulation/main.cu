#define _USE_MATH_DEFINES // for C++
#include <cmath>
#include <iostream>
#include <fstream>
#include <time.h>

#include <stdio.h>
#include <vector_types.h>
#include <cstdlib>

#define GL_GLEXT_PROTOTYPES
//#include <GL/glew.h>
#include <GL/glut.h>


#include <cuda_gl_interop.h>
#include <device_launch_parameters.h>

//utility functions declarations go here

// defined the key board function
void keyboardFunc(unsigned char key, int x, int y);

// defined call back function triggered by mouse
void mouseCallback(int x, int y);

// defined callback function for resizing the view
void resizeCallback(int w, int h);

// for the last mouse motion
void PassiveMouseMotion(int x, int y);

void timerFunc(int value);

void draw2(void);

void cross(float x1, float y1, float z1, float x2, float y2, float z2, float& rightX, float& rightY, float& rightZ);

const int ORTHO_VERSION = 0; // 1 is 2D version, 0 is 3D version.

#define WINDOW_W 1920
#define WINDOW_H 1080

#define M_PI       3.14159265358979323846   // pi

#define N_SIZE 10000
#define BLOCK_SIZE 512 //If you put a block size higher than the maximum block size supported by the GPU your kernel wont do anything when launched (see deviceQuery)
#define GRID_SIZE 1000
#define SOFT_FACTOR 0.00125f

#define GRAVITATIONAL_CONSTANT 0.01f
#define TIME_STEP 0.001f
#define PI 3.14152926f
#define DENSITY 1000000


extern float3 pos[N_SIZE];
extern float m[N_SIZE];
extern float r[N_SIZE];

struct Camera {
    float camX, camY, camZ;
    float forwardX, forwardY, forwardZ;
    float upX, upY, upZ;

    float theta, phi;

    Camera() {
        camX = 0, camY = 0, camZ = 200;
        forwardX = 0, forwardY = 0, forwardZ = -1;
        upX = 0, upY = 1, upZ = 0;

        theta = 0; phi = M_PI;
    }
};
Camera camera;
extern GLuint vertexArray;
extern float cx, cy, cz;


void init();
void deinit();


void initCUDA();
void initGL();


int runKernelNBodySimulation();

__global__
void nbody(float3* pos, float3* acc, float3* vel, float* m, float* r);

int bodies_size_float3 = 0;
int bodies_size_float = 0;
float3* pos_dev = NULL;
float3* vel_dev = NULL;
float3* acc_dev = NULL;
float* m_dev = NULL;
float* r_dev = NULL;

float3 pos[N_SIZE];
float3 vel[N_SIZE];
float3 acc[N_SIZE];
float m[N_SIZE];
float r[N_SIZE];

// Camera camera;

GLuint vertexArray;

__device__
int icbrt2(unsigned x) {
	int s;
	unsigned y, b, y2;

	y2 = 0;
	y = 0;
	for (s = 30; s >= 0; s = s - 3) {
		y2 = 4 * y2;
		y = 2 * y;
		b = (3 * (y2 + y) + 1) << s;
		if (x >= b) {
			x = x - b;
			y2 = y2 + 2 * y + 1;
			y = y + 1;
		}
	}
	return y;
}

void initBody(int i) {

	if (ORTHO_VERSION) {
		pos[i].x = (-WINDOW_W / 2 + ((float)rand() / (float)(RAND_MAX)) * WINDOW_W) * 0.9;
		pos[i].y = (-WINDOW_H / 2 + ((float)rand() / (float)(RAND_MAX)) * WINDOW_H) * 0.9;
		pos[i].z = 0.0f;

		acc[i].x = -5 + ((float)rand() / (float)(RAND_MAX)) * 10;
		acc[i].y = -5 + ((float)rand() / (float)(RAND_MAX)) * 10;
		acc[i].z = -5 + ((float)rand() / (float)(RAND_MAX)) * 10;

		vel[i].x = -5 + ((float)rand() / (float)(RAND_MAX)) * 10;
		vel[i].y = -5 + ((float)rand() / (float)(RAND_MAX)) * 10;
		vel[i].z = -5 + ((float)rand() / (float)(RAND_MAX)) * 10;

	}
	else {
		pos[i].x = (-WINDOW_W / 2 + ((float)rand() / (float)(RAND_MAX)) * WINDOW_W) * 0.9;
		pos[i].y = (-WINDOW_H / 2 + ((float)rand() / (float)(RAND_MAX)) * WINDOW_H) * 0.9;
		pos[i].z = (-500 + ((float)rand() / (float)(RAND_MAX)) * 500) * 0.9;

		acc[i].x = -50 + ((float)rand() / (float)(RAND_MAX)) * 50;
		acc[i].y = -50 + ((float)rand() / (float)(RAND_MAX)) * 50;
		acc[i].z = -50 + ((float)rand() / (float)(RAND_MAX)) * 50;

		vel[i].x = -50 + ((float)rand() / (float)(RAND_MAX)) * 50;
		vel[i].y = -50 + ((float)rand() / (float)(RAND_MAX)) * 50;
		vel[i].z = -50 + ((float)rand() / (float)(RAND_MAX)) * 50;
	}

	r[i] = ((float)rand() / (float)(RAND_MAX)) * 3.0;
	m[i] = 4.0 / 3.0 * PI * r[i] * r[i] * r[i] * DENSITY;
}

void initCUDA()
{


	bodies_size_float3 = N_SIZE * sizeof(float3);
	bodies_size_float = N_SIZE * sizeof(float);

	cudaMalloc((void**)&pos_dev, bodies_size_float3);
	cudaMalloc((void**)&acc_dev, bodies_size_float3);
	cudaMalloc((void**)&vel_dev, bodies_size_float3);
	cudaMalloc((void**)&m_dev, bodies_size_float);
	cudaMalloc((void**)&r_dev, bodies_size_float);

	for (int i = 0; i < N_SIZE; i++) {
		initBody(i);
	}


	cudaMemcpy(pos_dev, pos, bodies_size_float3, cudaMemcpyHostToDevice);
	cudaMemcpy(acc_dev, acc, bodies_size_float3, cudaMemcpyHostToDevice);
	cudaMemcpy(vel_dev, vel, bodies_size_float3, cudaMemcpyHostToDevice);
	cudaMemcpy(m_dev, m, bodies_size_float, cudaMemcpyHostToDevice);
	cudaMemcpy(r_dev, r, bodies_size_float, cudaMemcpyHostToDevice);

}

void initGL()
{

	glEnable(GL_CULL_FACE);
	//glEnable(GL_VERTEX_PROGRAM_POINT_SIZE);

	glEnable(GL_LIGHTING);
	glLightModeli(GL_LIGHT_MODEL_LOCAL_VIEWER, GL_TRUE);

	glEnable(GL_LIGHT0);
	glEnable(GL_COLOR_MATERIAL);

	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	/*void glOrtho(GLdouble  left,  GLdouble  right,  GLdouble  bottom,  GLdouble  top,  GLdouble  nearVal,  GLdouble  farVal);*/
	if (ORTHO_VERSION)
	{
		glOrtho(-WINDOW_W / 2, WINDOW_W / 2, -WINDOW_H / 2, WINDOW_H / 2, -100, 100);
	}
	else
	{
		gluPerspective(45, (float)WINDOW_W / (float)WINDOW_H, 1, 2000);
	}

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	if (!ORTHO_VERSION)
		gluLookAt(camera.camX, camera.camY, camera.camZ, //Camera position
			camera.camX + camera.forwardX, camera.camY + camera.forwardY, camera.camZ + camera.forwardZ, //Position of the object to look at
			camera.upX, camera.upY, camera.upZ); //Camera up direction


	glEnable(GL_DEPTH_TEST);
	glEnable(GL_FOG);

}

// init the program
void init()
{
	initGL();
	initCUDA();
	atexit(deinit);
}

void deinit()
{
	cudaFree(pos_dev);
	cudaFree(r_dev);
	cudaFree(m_dev);
	cudaFree(acc_dev);
	cudaFree(vel_dev);
}


/**
 * update the position and velocity for each star
 */

__device__
void updatePosAndVel(float3 pos[], float3 vel[], float3 acc[], float3 cur_a, int self)
{
	float newvx = vel[self].x + (acc[self].x + cur_a.x) / 2 * TIME_STEP;
	float newvy = vel[self].y + (acc[self].y + cur_a.y) / 2 * TIME_STEP;
	float newvz = vel[self].z + (acc[self].z + cur_a.z) / 2 * TIME_STEP;

	//update position
	pos[self].x += newvx * TIME_STEP + acc[self].x * TIME_STEP * TIME_STEP / 2;
	pos[self].y += newvy * TIME_STEP + acc[self].y * TIME_STEP * TIME_STEP / 2;
	pos[self].z += newvz * TIME_STEP + acc[self].z * TIME_STEP * TIME_STEP / 2;

	//update velocity
	vel[self].x = newvx;
	vel[self].y = newvy;
	vel[self].z = newvz;
}

/**
 * the accelartion on one star if they do not collide
 */
__device__
void bodyBodyInteraction(float3& acc, float m[], int self, int other, float3 dist3, float dist_sqr)
{

	float dist_six = dist_sqr * dist_sqr * dist_sqr;
	float dist_cub = sqrtf(dist_six);

	// this is according to the Newton's law of universal gravitaion
	acc.x += (m[other] * dist3.x) / dist_cub;
	acc.y += (m[other] * dist3.y) / dist_cub;
	acc.z += (m[other] * dist3.z) / dist_cub;
}

/**
 * the accelartion on one star if they collide
 */
__device__
void bodyBodyCollision(float m[], float3 vel[], float3 acc[], int self, int other)
{

	float mass = m[self] + m[other];

	// Used perfectly unelastic collision model to caculate the velocity after merging.
	float3 velocity;

	velocity.x = (vel[self].x * m[self] + vel[other].x * m[other]) / mass;
	velocity.y = (vel[self].y * m[self] + vel[other].y * m[other]) / mass;
	velocity.z = (vel[self].z * m[self] + vel[other].z * m[other]) / mass;

	float3 zero_float3;
	zero_float3.x = 0.0f;
	zero_float3.y = 0.0f;
	zero_float3.z = 0.0f;

	acc[self] = zero_float3;
	acc[other] = zero_float3;

	// the heavier body will remain, but the lighter one will disappear
	// although here will cause code divergence, the 4 operations are very simple
	if (m[self] > m[other])
	{
		vel[self] = velocity;
		vel[other] = zero_float3;
		m[self] = mass;
		m[other] = 0.0f;
	}
	else
	{
		vel[other] = velocity;
		vel[self] = zero_float3;
		m[other] = mass;
		m[self] = 0.0f;
	}
}


__global__
void nbody(float3* pos, float3* acc, float3* vel, float* m, float* r)
{
	int idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;

	if (idx < N_SIZE && m[idx] != 0)
	{
		float m_before = m[idx];

		// initiate the acceleration of the next moment 
		float3 cur_acc;

		cur_acc.x = 0;
		cur_acc.y = 0;
		cur_acc.z = 0;

		// for any two body
		for (int i = 0; i < N_SIZE; i++) {

			if (i != idx && m[i] != 0) {

				if (m[idx] != 0) {

					float3 dist3; // calculate their distance

					dist3.x = pos[i].x - pos[idx].x;
					dist3.y = pos[i].y - pos[idx].y;
					dist3.z = pos[i].z - pos[idx].z;

					// update the force between two non-empty bodies
					float dist_sqr = dist3.x * dist3.x + dist3.y * dist3.y + dist3.z * dist3.z + SOFT_FACTOR;

					// if they depart
					if (sqrt(dist_sqr) > r[idx] + r[i])
						bodyBodyInteraction(cur_acc, m, idx, i, dist3, dist_sqr);

					// if they overlap
					else
						bodyBodyCollision(m, vel, acc, idx, i);

				}
			}
		}

		// multiplies a Gravitational Constant
		cur_acc.x *= GRAVITATIONAL_CONSTANT;
		cur_acc.y *= GRAVITATIONAL_CONSTANT;
		cur_acc.z *= GRAVITATIONAL_CONSTANT;

		//update the position and velocity
		updatePosAndVel(pos, vel, acc, cur_acc, idx);

		// update the body acceleration
		acc[idx].x = cur_acc.x;
		acc[idx].y = cur_acc.y;
		acc[idx].z = cur_acc.z;

		// if the mass is changed, update the radius
		if (m[idx] != m_before)
			r[idx] = icbrt2(m[idx] / (DENSITY * 4.0 / 3.0 * PI));
	}
}



int runKernelNBodySimulation()
{
	// Map the buffer to CUDA

	nbody << <GRID_SIZE, BLOCK_SIZE >> > (pos_dev, acc_dev, vel_dev, m_dev, r_dev);

	cudaMemcpy(pos, pos_dev, bodies_size_float3, cudaMemcpyDeviceToHost);
	cudaMemcpy(m, m_dev, bodies_size_float, cudaMemcpyDeviceToHost);
	cudaMemcpy(r, r_dev, bodies_size_float, cudaMemcpyDeviceToHost);
	return EXIT_SUCCESS;
}

float prevX = WINDOW_W / 2, prevY = WINDOW_H / 2;
bool mouseUp = 0;
bool toggleHelp = true;

extern float3 pos[N_SIZE];
extern float3 vel[N_SIZE];
extern float3 acc[N_SIZE];
extern float m[N_SIZE];
extern float r[N_SIZE];

GLfloat lpos[4] = { -0.3,0.0,200,0 }; //Positioned light
GLfloat light_specular[4] = { 1, 0.6, 1, 0 }; //specular light intensity (color)
GLfloat light_diffuse[] = { 1.0, 1.0, 1.0, 0.0 };//diffuse light intensity (color)
GLfloat light_ambient[] = { 0.2, 0.2, 0.2, 0.0 }; //ambient light intensity (color)
GLfloat a;
GLfloat mat_emission[] = { 0.8, 0.5, 0.3, 0.0 }; //object material preperty emission of light
GLfloat mat_specular[] = { 4.0, 0.5, 2.0, 0.0 }; //object material specularity
GLfloat low_shininess[] = { 50 };
GLfloat fogColor[] = { 0.5f, 0.5f, 0.5f, 1 };

void timerFunc(int value)
{
    glutPostRedisplay();
}

void resizeCallback(int w, int h) {
    if (ORTHO_VERSION) return;
    // Prevent a divide by zero, when window is too short
    // (you cant make a window of zero width).
    if (h == 0)
        h = 1;

    float ratio = 1.0 * w / h;

    // Reset the coordinate system before modifying
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();

    // Set the viewport to be the entire window
    glViewport(0, 0, w, h);

    // Set the correct perspective.
    gluPerspective(45, ratio, 1, 1000);
    glMatrixMode(GL_MODELVIEW);

}


void keyboardFunc(unsigned char key, int x, int y) {

    if (key == 27)
        exit(0);

    float vel = 5.0;
    float rightX, rightY, rightZ;
    cross(camera.forwardX, camera.forwardY, camera.forwardZ, camera.upX, camera.upY, camera.upZ, rightX, rightY, rightZ);
    float sizeRight = sqrtf(rightX * rightX + rightY * rightY + rightZ * rightZ);
    rightX /= sizeRight; rightY /= sizeRight; rightZ /= sizeRight;

    if (key == 'w') // move forward
    {
        camera.camX += camera.forwardX * vel;
        camera.camY += camera.forwardY * vel;
        camera.camZ += camera.forwardZ * vel;
    }
    if (key == 's') // move backward
    {
        camera.camX -= camera.forwardX * vel;
        camera.camY -= camera.forwardY * vel;
        camera.camZ -= camera.forwardZ * vel;
    }
    if (key == 'a') // move left
    {

        camera.camX -= rightX * vel;
        camera.camY -= rightY * vel;
        camera.camZ -= rightZ * vel;
    }
    if (key == 'd') // move right
    {
        camera.camX += rightX * vel;
        camera.camY += rightY * vel;
        camera.camZ += rightZ * vel;
    }

    if (key == 'h') // show or hide help
    {
        toggleHelp = !toggleHelp;
    }


}
void PassiveMouseMotion(int x, int y) {
    prevX = x, prevY = y;

}

// call back function triggered by mouse
void mouseCallback(int x, int y) {

    float velx = (float(x - prevX) / WINDOW_W);
    float vely = (float(y - prevY) / WINDOW_H);
    prevX = x;
    prevY = y;
    camera.phi += -velx * M_PI * 0.9;
    camera.theta += -vely * M_PI * 0.9;

    float rightX, rightY, rightZ;
    rightX = sinf(camera.phi - M_PI / 2.0f);
    rightY = 0;
    rightZ = cosf(camera.phi - M_PI / 2.0f);
    float sizeRight = sqrtf(rightX * rightX + rightY * rightY + rightZ * rightZ);
    rightX /= sizeRight; rightY /= sizeRight; rightZ /= sizeRight;


    camera.forwardX = cosf(camera.theta) * sinf(camera.phi);
    camera.forwardY = sinf(camera.theta);
    camera.forwardZ = cosf(camera.theta) * cosf(camera.phi);

    float sizeForward = sqrtf(camera.forwardX * camera.forwardX + camera.forwardY * camera.forwardY + camera.forwardZ * camera.forwardZ);
    camera.forwardX /= sizeForward; camera.forwardY /= sizeForward; camera.forwardZ /= sizeForward;

    float newUpX, newUpY, newUpZ;

    cross(rightX, rightY, rightZ, camera.forwardX, camera.forwardY, camera.forwardZ, newUpX, newUpY, newUpZ);
    float sizeUp = sqrtf(newUpX * newUpX + newUpY * newUpY + newUpZ * newUpZ);
    camera.upX = newUpX / sizeUp; camera.upY = newUpY / sizeUp; camera.upZ = newUpZ / sizeUp;



}

void cross(float x1, float y1, float z1, float x2, float y2, float z2, float& rightX, float& rightY, float& rightZ) {
    rightX = y1 * z2 - z1 * y2;
    rightY = x1 * z2 - x1 * z2;
    rightZ = x1 * y2 - y1 * x1;

}



void DrawCircle(float cx, float cy, float r, int num_segments) {
    glBegin(GL_LINE_LOOP);
    for (int ii = 0; ii < num_segments; ii++) {
        float theta = 2.0f * M_PI * float(ii) / float(num_segments);//get the current angle 
        float x = r * cosf(theta);//calculate the x component 
        float y = r * sinf(theta);//calculate the y component 
        glVertex2f(x + cx, y + cy);//output vertex 
    }
    glEnd();
}

void drawText(std::string text, float x, float y) {
    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();


    glColor3f(1.0f, 0.0f, 0.0f);//needs to be called before RasterPos
    glRasterPos2f(x, y);


    void* font = GLUT_BITMAP_TIMES_ROMAN_24;

    for (std::string::iterator i = text.begin(); i != text.end(); ++i)
    {
        char c = *i;
        glutBitmapCharacter(font, c);
    }
    glPopMatrix();
}


void setLights() {


    glMaterialfv(GL_FRONT, GL_EMISSION, mat_emission);
    glMaterialfv(GL_FRONT, GL_SPECULAR, mat_specular);
    glMaterialfv(GL_FRONT, GL_SHININESS, low_shininess);
    glLightfv(GL_LIGHT0, GL_SPECULAR, light_specular);
    glLightfv(GL_LIGHT0, GL_POSITION, lpos);


    //Adding fog
    glFogfv(GL_FOG_COLOR, fogColor);
    glFogi(GL_FOG_MODE, GL_LINEAR);
    glFogf(GL_FOG_START, 10.0f);
    glFogf(GL_FOG_END, 1000.0f);
}

void draw2() {
    glClearColor(0.1f, 0.1f, 0.1f, 0.1f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);


    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    if (!ORTHO_VERSION) {
        gluLookAt(camera.camX, camera.camY, camera.camZ, //Camera position
            camera.camX + camera.forwardX, camera.camY + camera.forwardY, camera.camZ + camera.forwardZ, //Position of the object to look at
            camera.upX, camera.upY, camera.upZ); //Camera up direction
    }
    setLights();

    runKernelNBodySimulation();

    if (toggleHelp) {
        if (!ORTHO_VERSION)
        {
            drawText("USAGE INFO", 50, 60);
            drawText("Use keys w, a, s, d to move", 50, 50);
            drawText("Hold the left button on the mouse to look around", 50, 40);
            drawText("Press h to show/hide this help info", 50, 30);
        }
        else
        {
            drawText("USAGE INFO", 50, 80);
            drawText("Use keys W, A, S, D to move", 50, 60);
            drawText("Hold the left button on the mouse to look around", 50, 40);
            drawText("Press H to show/hide this help info", 50, 10);
        }
    }


    glColor3f(0.5f, 0.5f, 0.3f);
    for (int i = 0; i < N_SIZE; i++) {
        if (m[i] > 0)
        {
            if (!ORTHO_VERSION)
            {
                glPushMatrix();
                glTranslatef(pos[i].x, pos[i].y, pos[i].z);
                glutSolidSphere(r[i], 10, 10); // draw sphere
                glPopMatrix();
            }
            else {
                DrawCircle(pos[i].x, pos[i].y, r[i], 10); // draw circle
            }
        }

    }

    glutSwapBuffers();


}

//Main program
int main(int argc, char** argv) {

    srand(time(NULL));

    glutInit(&argc, argv);

    /*Setting up  The Display
    /    -RGB color model + Alpha Channel = GLUT_RGBA
    */
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH);

    //Configure Window Postion
    glutInitWindowPosition(50, 25);

    //Configure Window Size
    glutInitWindowSize(WINDOW_W, WINDOW_H);

    //Create Window
    glutCreateWindow("NBody Simulation");

    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glEnable(GL_BLEND);


    //Call to the drawing function
    glutDisplayFunc(draw2);
    glutIdleFunc(draw2);
    //glutTimerFunc(1000,timerFunc,10);

    glutKeyboardFunc(keyboardFunc);
    glutMotionFunc(mouseCallback);
    glutPassiveMotionFunc(PassiveMouseMotion);
    glutReshapeFunc(resizeCallback);


    // init the CUDA and the OpenGL
    init();

    // Loop require by OpenGL
    glutMainLoop();


    return 0;
}
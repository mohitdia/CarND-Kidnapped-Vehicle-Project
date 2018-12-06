/*
 * particle_filter.cpp
 *
 *  Modified on: Dec 06, 2018
 *      Author: Mohit Dia
 * 
 * References: https://github.com/JunshengFu/kidnapped-vehicle
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

static int NUM_PARTICLES = 100;
default_random_engine gen;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
  
  // create normal distributions for x, y, and theta
  normal_distribution<double> dist_x(x, std[0]);
  normal_distribution<double> dist_y(y, std[1]);
  normal_distribution<double> dist_theta(theta, std[2]);

  // Initializing number of particles and resizeing the vectors of particles
  num_particles = NUM_PARTICLES;
  particles.resize(num_particles);

  // generate the particles
  for(auto& p: particles){
    p.x = dist_x(gen);
    p.y = dist_y(gen);
    p.theta = dist_theta(gen);
    p.weight = 1;
  }

  is_initialized = true;

}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/

  // generate random Gaussian noise
  normal_distribution<double> dist_x(0, std_pos[0]);
  normal_distribution<double> dist_y(0, std_pos[1]);
  normal_distribution<double> dist_theta(0, std_pos[2]);

  for(auto& p: particles){

    // update particle's measurements
    if( fabs(yaw_rate) < 0.0001){
      p.x += velocity * delta_t * cos(p.theta);
      p.y += velocity * delta_t * sin(p.theta);

    } else{
      p.x += velocity / yaw_rate * ( sin( p.theta + yaw_rate*delta_t ) - sin(p.theta) );
      p.y += velocity / yaw_rate * ( cos( p.theta ) - cos( p.theta + yaw_rate*delta_t ) );
      p.theta += yaw_rate * delta_t;
    }

    // Add sensor noise
    p.x += dist_x(gen);
    p.y += dist_y(gen);
    p.theta += dist_theta(gen);
  }

}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.

  for(auto& obs: observations){

    // Initialize min distance to numeric max limit for floats
    double minD = std::numeric_limits<float>::max();

    for(const auto& pred: predicted){
      double distance = dist(obs.x, obs.y, pred.x, pred.y);
      if( minD > distance){
        minD = distance;
        obs.id = pred.id;
      }
    }
  }
}

/* **
 * This method stores the map landmarks which are within the sensor range with respect to the particle.
 * */
vector<LandmarkObs> gatherNearestLandMarks(double sensor_range, Particle p, const Map &map_landmarks) {
  vector<LandmarkObs> predictions;

  for(const auto& lm: map_landmarks.landmark_list){
      double distance = dist(p.x, p.y, lm.x_f, lm.y_f);
      if( distance < sensor_range){
        predictions.push_back(LandmarkObs{lm.id_i, lm.x_f, lm.y_f});
      }
    }
  
  return predictions;
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
  // TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html
  for(auto& p: particles){
    p.weight = 1.0;

    // Step 1: Gather landmarks which are within sensor range
    vector<LandmarkObs> predictions = gatherNearestLandMarks(sensor_range, p, map_landmarks);

    // Step 2: Transform observations coordinates from vehicle to map coordinates.
    vector<LandmarkObs> transformed_observations;
    double cos_theta = cos(p.theta);
    double sin_theta = sin(p.theta);

    for(const auto& obs: observations){
      LandmarkObs tmp;
      tmp.x = obs.x * cos_theta - obs.y * sin_theta + p.x;
      tmp.y = obs.x * sin_theta + obs.y * cos_theta + p.y;
      //tmp.id = obs.id; // maybe an unnecessary step, since the each obersation will get the id from dataAssociation step.
      transformed_observations.push_back(tmp);
    }

    // Step 3: Find landmark index for each observation using nearest neighbors.
    dataAssociation(predictions, transformed_observations);

    // Step 4: Compute the particle's weight:
    for(const auto& obs_m: transformed_observations){

      Map::single_landmark_s landmark = map_landmarks.landmark_list.at(obs_m.id-1);
      double x_term = pow(obs_m.x - landmark.x_f, 2) / (2 * pow(std_landmark[0], 2));
      double y_term = pow(obs_m.y - landmark.y_f, 2) / (2 * pow(std_landmark[1], 2));
      double w = exp(-(x_term + y_term)) / (2 * M_PI * std_landmark[0] * std_landmark[1]);
      p.weight *=  w;
    }

    weights.push_back(p.weight);

  }

}

void ParticleFilter::resample() {
// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

  vector<Particle> updated_particles;

  // get all of the current weights
  vector<double> weights;
  for (int i = 0; i < num_particles; i++) {
    weights.push_back(particles[i].weight);
  }

  // Generate random starting index for resampling wheel
  uniform_int_distribution<int> uniintdist(0, num_particles-1);
  auto index = uniintdist(gen);

  double max_weight = *max_element(weights.begin(), weights.end());

  // uniform random distribution [0.0, max_weight)
  uniform_real_distribution<double> unirealdist(0.0, max_weight);

  double beta = 0.0;

  // Finishing resampling using sampling wheel technique
  for (int i = 0; i < num_particles; i++) {
    beta += unirealdist(gen) * 2.0;
    while (beta > weights[index]) {
      beta -= weights[index];
      index = (index + 1) % num_particles;
    }
    updated_particles.push_back(particles[index]);
  }

  particles = updated_particles;
}

// Particle ParticleFilter::SetAssociations(Particle &particle, const std::vector<int> &associations,
//                                          const std::vector<double> &sense_x, const std::vector<double> &sense_y)
// {
//   //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
//   // associations: The landmark id that goes along with each listed association
//   // sense_x: the associations x mapping already converted to world coordinates
//   // sense_y: the associations y mapping already converted to world coordinates

//   particle.associations = associations;
//   particle.sense_x = sense_x;
//   particle.sense_y = sense_y;
// }

string ParticleFilter::getAssociations(Particle best)
{
  vector<int> v = best.associations;
  stringstream ss;
  copy(v.begin(), v.end(), ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length() - 1); // get rid of the trailing space
  return s;
}
string ParticleFilter::getSenseX(Particle best)
{
  vector<double> v = best.sense_x;
  stringstream ss;
  copy(v.begin(), v.end(), ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length() - 1); // get rid of the trailing space
  return s;
}
string ParticleFilter::getSenseY(Particle best)
{
  vector<double> v = best.sense_y;
  stringstream ss;
  copy(v.begin(), v.end(), ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length() - 1); // get rid of the trailing space
  return s;
}

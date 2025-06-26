<?php
	defined('BASEPATH') OR exit('No direct script access allowed');
	class Migration_ml_forecasting extends MY_Migration 
	{

	    public function up() 
			{
				$this->execute_sql(realpath(dirname(__FILE__).'/'.'20230905147456_ml_forecasting.sql'));
	    }

	    public function down() 
			{
	    }

	}
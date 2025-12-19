#pragma once


#include <thread>
#include <algorithm>
#include <msclr/gcroot.h>
#include "ImageProcessorWrapper.h"
#include "CudaProcessor.h"

#include <chrono>

namespace pavicgui2024 {

	using namespace System;
	using namespace System::ComponentModel;
	using namespace System::Collections;
	using namespace System::Windows::Forms;
	using namespace System::Data;
	using namespace System::Drawing;
	using namespace System::Drawing::Imaging;
	using namespace std::chrono;

	using namespace std;

	//================================================================================
	// // --- Início da função auxiliar para aplicar o filtro sépia parcialmente ---
	//// Esta função será executada pelas threads.
	//// Ela precisa ser definida fora da classe do formulário para ser usada com std::thread de forma mais simples.
	void ApplySepiaFilterPartial(Bitmap^ inputImage, Bitmap^ outputImage, int startY, int endY) {
		for (int i = 0; i < inputImage->Width; i++) {
			for (int j = startY; j < endY; j++) {
				// Passo 1: Obter a cor do pixel atual
				Color pixelColor = inputImage->GetPixel(i, j);

				// Passo 2: Extrair os valores originais de Vermelho, Verde e Azul
				int r = pixelColor.R;
				int g = pixelColor.G;
				int b = pixelColor.B;

				// Passo 3: Calcular os novos valores de pixel usando a Fórmula Sépia
				double tr = 0.393 * r + 0.769 * g + 0.189 * b;
				double tg = 0.349 * r + 0.686 * g + 0.168 * b;
				double tb = 0.272 * r + 0.534 * g + 0.131 * b;

				// Passo 4: Limitar os valores ao intervalo 0-255
				int newR = Math::Min(255, (int)tr);
				int newG = Math::Min(255, (int)tg);
				int newB = Math::Min(255, (int)tb);

				outputImage->SetPixel(i, j, Color::FromArgb(newR, newG, newB));
			}
		}
	}
	// --- Fim da função auxiliar ---
	// 
	// ==================================================================================
	//================================================================================
	// // --- Início da função auxiliar para aplicar o filtro sépia parcialmente ---
	//// Esta função será executada pelas threads.
	//// Ela precisa ser definida fora da classe do formulário para ser usada com std::thread de forma mais simples.
	void ApplySepiaFilterWindow(Bitmap^ inputImage, Bitmap^ outputImage, int startX, int endX, int startY, int endY) {
		for (int i = startX; i < endX; i++) {
			for (int j = startY; j < endY; j++) {
				// Passo 1: Obter a cor do pixel atual
				Color pixelColor = inputImage->GetPixel(i, j);

				// Passo 2: Extrair os valores originais de Vermelho, Verde e Azul
				int r = pixelColor.R;
				int g = pixelColor.G;
				int b = pixelColor.B;

				// Passo 3: Calcular os novos valores de pixel usando a Fórmula Sépia
				double tr = 0.393 * r + 0.769 * g + 0.189 * b;
				double tg = 0.349 * r + 0.686 * g + 0.168 * b;
				double tb = 0.272 * r + 0.534 * g + 0.131 * b;

				// Passo 4: Limitar os valores ao intervalo 0-255
				int newR = Math::Min(255, (int)tr);
				int newG = Math::Min(255, (int)tg);
				int newB = Math::Min(255, (int)tb);

				outputImage->SetPixel(i, j, Color::FromArgb(newR, newG, newB));
			}
		}
	}
	// --- Fim da função auxiliar ---
	// ===============================================================================
	// Helper struct to pass arguments to the unmanaged thread function
	struct SepiaThreadArgs {
		msclr::gcroot<Bitmap^> inputImage;
		msclr::gcroot<Bitmap^> outputImage;
		int startX;
		int endX;
		int startY;
		int endY;
	};

	// This function MUST be an unmanaged (native) C++ function
	// It cannot directly take Bitmap^ as parameters.
	void ApplySepiaFilterWindow_Unmanaged(SepiaThreadArgs* args) {
		Bitmap^ inputImage = args->inputImage;   // Get managed handle from gcroot
		Bitmap^ outputImage = args->outputImage; // Get managed handle from gcroot

		// Ensure the managed objects are still valid before using
		if (inputImage == nullptr || outputImage == nullptr) {
			// Handle error or return
			return;
		}

		// You would typically use LockBits here for performance with raw pixel data
		// because GetPixel/SetPixel are very slow and envolve interop calls per pixel.
		// For simplicity, keeping your original GetPixel/SetPixel logic:
		for (int i = args->startX; i < args->endX; i++) {
			for (int j = args->startY; j < args->endY; j++) {
				Color pixelColor = inputImage->GetPixel(i, j);

				int r = pixelColor.R;
				int g = pixelColor.G;
				int b = pixelColor.B;

				double tr = 0.393 * r + 0.769 * g + 0.189 * b;
				double tg = 0.349 * r + 0.686 * g + 0.168 * b;
				double tb = 0.272 * r + 0.534 * g + 0.131 * b;

				int newR = System::Math::Min(255, (int)tr); // Use System::Math for managed functions
				int newG = System::Math::Min(255, (int)tg);
				int newB = System::Math::Min(255, (int)tb);

				outputImage->SetPixel(i, j, Color::FromArgb(newR, newG, newB));
			}
		}

		// Don't delete args here if it was created on the stack of the calling function.
		// If it was dynamically allocated (e.g., `new SepiaThreadArgs()`), then you'd delete it here.
	}

	// This function MUST be an unmanaged (native) C++ function
	// It cannot directly take Bitmap^ as parameters.
	void ApplyBWFilterWindow_Unmanaged(SepiaThreadArgs* args) {
		Bitmap^ inputImage = args->inputImage;   // Get managed handle from gcroot
		Bitmap^ outputImage = args->outputImage; // Get managed handle from gcroot

		// Ensure the managed objects are still valid before using
		if (inputImage == nullptr || outputImage == nullptr) {
			// Handle error or return
			return;
		}

		//// Apply the black and white filter
		// You would typically use LockBits here for performance with raw pixel data
		// because GetPixel/SetPixel are very slow and envolve interop calls per pixel.
		// For simplicity, keeping your original GetPixel/SetPixel logic:
		for (int i = args->startX; i < args->endX; i++) {
			for (int j = args->startY; j < args->endY; j++) {
				
				Color pixelColor = inputImage->GetPixel(i, j);
				int grayValue = (int)(0.299 * pixelColor.R + 0.587 * pixelColor.G + 0.114 * pixelColor.B);
				outputImage->SetPixel(i, j, Color::FromArgb(grayValue, grayValue, grayValue));
				
			}
		}
		

		// Don't delete args here if it was created on the stack of the calling function.
		// If it was dynamically allocated (e.g., `new SepiaThreadArgs()`), then you'd delete it here.
	}

	// 
	// ===============================================================================
	// 
	//================================================================================
	//================================================================================
	// --- Início da função auxiliar para aplicar o filtro sépia parcialmente ---
	// Esta função será executada pelas threads e operará em dados brutos.
	// Ela precisa ser definida fora da classe do formulário.
	void ApplySepiaFilterPartialRaw(IntPtr inputScan0, int inputStride, IntPtr outputScan0, int outputStride, int width, int bytesPerPixel, int startY, int endY) {
		// Converter IntPtr para ponteiros de bytes não gerenciados
		unsigned char* ptrInput = (unsigned char*)inputScan0.ToPointer();
		unsigned char* ptrOutput = (unsigned char*)outputScan0.ToPointer();

		for (int j = startY; j < endY; j++) {
			for (int i = 0; i < width; i++) {
				// Calcular o offset para o pixel atual na linha
				long offsetInput = (long)j * inputStride + (long)i * bytesPerPixel;
				long offsetOutput = (long)j * outputStride + (long)i * bytesPerPixel;

				// Obter os valores de cor (assumindo formato BGR ou BGRA)
				int b = ptrInput[offsetInput];
				int g = ptrInput[offsetInput + 1];
				int r = ptrInput[offsetInput + 2];
				// Se for 32bpp (BGRA), o quarto byte é o canal alfa.
				// int a = (bytesPerPixel == 4) ? ptrInput[offsetInput + 3] : 255; // Linha comentada por enquanto, se precisar de alfa, descomente e ajuste

				// Aplicar a fórmula Sépia
				double tr = 0.393 * r + 0.769 * g + 0.189 * b;
				double tg = 0.349 * r + 0.686 * g + 0.168 * b;
				double tb = 0.272 * r + 0.534 * g + 0.131 * b;

				// Limitar os valores ao intervalo 0-255
				int newR = Math::Min(255, (int)tr);
				int newG = Math::Min(255, (int)tg);
				int newB = Math::Min(255, (int)tb);

				// Definir os novos valores de pixel na imagem de saída
				ptrOutput[offsetOutput] = (unsigned char)newB;
				ptrOutput[offsetOutput + 1] = (unsigned char)newG;
				ptrOutput[offsetOutput + 2] = (unsigned char)newR;
				// Se for 32bpp (BGRA), manter o canal alfa original ou definir como 255
				// if (bytesPerPixel == 4) ptrOutput[offsetOutput + 3] = (unsigned char)a; // Linha comentada por enquanto, se precisar de alfa, descomente e ajuste
			}
		}
	}
	// --- Fim da função auxiliar ---
	//==============================================================


	/// <summary>
	/// Summary for Pavic_gui_2024_Form
	/// </summary>
	public ref class Pavic_gui_2024_Form : public System::Windows::Forms::Form
	{
	private:
		ImageProcessorWrapper^ processor;
		System::Windows::Forms::Button^ btInvertCuda;

	public:
		Pavic_gui_2024_Form(void)
		{
			InitializeComponent();
			processor = gcnew ImageProcessorWrapper();
			// processor = gcnew ImageProcessorWrapper();
		}

	protected:
		/// <summary>
		/// Clean up any resources being used.
		/// </summary>
		~Pavic_gui_2024_Form()
		{
			if (components)
			{
				delete components;
			}
		}
	private: System::Windows::Forms::Button^ bt_open;
	protected:
	private: System::Windows::Forms::Button^ bt_close;
	private: System::Windows::Forms::Button^ bt_exit;
	private: System::Windows::Forms::PictureBox^ pbox_input;

	private: System::Windows::Forms::PictureBox^ pbox_output;
	private: System::Windows::Forms::Button^ bt_copy;
	private: System::Windows::Forms::Button^ bt_filter_bw;
	private: System::Windows::Forms::Label^ label1;
	private: System::Windows::Forms::Label^ label2;

	private: System::Windows::Forms::Button^ bt_close_output;

	private: System::Windows::Forms::Label^ label11;
	private: System::Windows::Forms::Label^ label12;
	private: System::Diagnostics::Stopwatch^ copyStopwatch;
	private: System::Diagnostics::Stopwatch^ filterStopwatch;
	private: System::Windows::Forms::Button^ bt_filter_Sepia;
	private: System::Windows::Forms::Button^ bt_filter_Sepia_MultiThread;
	private: System::Windows::Forms::Button^ bt_filter_Sepia_top;



	private: System::Windows::Forms::Label^ lb_timer;
	private: System::Windows::Forms::TextBox^ textB_Time;



	private: System::Windows::Forms::Button^ btCuda;

		System::ComponentModel::Container ^components;

#pragma region Windows Form Designer generated code
		/// <summary>
		/// Required method for Designer support - do not modify
		/// the contents of this method with the code editor.
		/// </summary>
		void InitializeComponent(void)
		{
			this->bt_open = (gcnew System::Windows::Forms::Button());
			this->bt_close = (gcnew System::Windows::Forms::Button());
			this->bt_exit = (gcnew System::Windows::Forms::Button());
			this->pbox_input = (gcnew System::Windows::Forms::PictureBox());
			this->pbox_output = (gcnew System::Windows::Forms::PictureBox());
			this->bt_copy = (gcnew System::Windows::Forms::Button());
			this->bt_filter_bw = (gcnew System::Windows::Forms::Button());
			this->label1 = (gcnew System::Windows::Forms::Label());
			this->label2 = (gcnew System::Windows::Forms::Label());
			this->bt_close_output = (gcnew System::Windows::Forms::Button());
			this->bt_filter_Sepia = (gcnew System::Windows::Forms::Button());
			this->bt_filter_Sepia_MultiThread = (gcnew System::Windows::Forms::Button());
			this->bt_filter_Sepia_top = (gcnew System::Windows::Forms::Button());
			this->lb_timer = (gcnew System::Windows::Forms::Label());
			this->textB_Time = (gcnew System::Windows::Forms::TextBox());
			this->btCuda = (gcnew System::Windows::Forms::Button());
			this->btInvertCuda = (gcnew System::Windows::Forms::Button());
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->pbox_input))->BeginInit();
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->pbox_output))->BeginInit();
			this->SuspendLayout();
			// 
			// bt_open
			// 
			this->bt_open->Location = System::Drawing::Point(9, 10);
			this->bt_open->Margin = System::Windows::Forms::Padding(2);
			this->bt_open->Name = L"bt_open";
			this->bt_open->Size = System::Drawing::Size(142, 37);
			this->bt_open->TabIndex = 0;
			this->bt_open->Text = L"Open";
			this->bt_open->UseVisualStyleBackColor = true;
			this->bt_open->Click += gcnew System::EventHandler(this, &Pavic_gui_2024_Form::bt_open_Click);
			// 
			// bt_close
			// 
			this->bt_close->Location = System::Drawing::Point(291, 192);
			this->bt_close->Margin = System::Windows::Forms::Padding(2);
			this->bt_close->Name = L"bt_close";
			this->bt_close->Size = System::Drawing::Size(95, 28);
			this->bt_close->TabIndex = 1;
			this->bt_close->Text = L"Close";
			this->bt_close->UseVisualStyleBackColor = true;
			this->bt_close->Click += gcnew System::EventHandler(this, &Pavic_gui_2024_Form::bt_close_Click);
			// 
			// bt_exit
			// 
			this->bt_exit->Location = System::Drawing::Point(9, 93);
			this->bt_exit->Margin = System::Windows::Forms::Padding(2);
			this->bt_exit->Name = L"bt_exit";
			this->bt_exit->Size = System::Drawing::Size(142, 37);
			this->bt_exit->TabIndex = 2;
			this->bt_exit->Text = L"Exit";
			this->bt_exit->UseVisualStyleBackColor = true;
			this->bt_exit->Click += gcnew System::EventHandler(this, &Pavic_gui_2024_Form::bt_exit_Click);
			// 
			// pbox_input
			// 
			this->pbox_input->BorderStyle = System::Windows::Forms::BorderStyle::Fixed3D;
			this->pbox_input->Location = System::Drawing::Point(13, 225);
			this->pbox_input->Margin = System::Windows::Forms::Padding(2);
			this->pbox_input->Name = L"pbox_input";
			this->pbox_input->Size = System::Drawing::Size(374, 331);
			this->pbox_input->SizeMode = System::Windows::Forms::PictureBoxSizeMode::StretchImage;
			this->pbox_input->TabIndex = 3;
			this->pbox_input->TabStop = false;
			this->pbox_input->Click += gcnew System::EventHandler(this, &Pavic_gui_2024_Form::pbox_input_Click);
			// 
			// pbox_output
			// 
			this->pbox_output->BorderStyle = System::Windows::Forms::BorderStyle::Fixed3D;
			this->pbox_output->Location = System::Drawing::Point(455, 225);
			this->pbox_output->Margin = System::Windows::Forms::Padding(2);
			this->pbox_output->Name = L"pbox_output";
			this->pbox_output->Size = System::Drawing::Size(374, 331);
			this->pbox_output->SizeMode = System::Windows::Forms::PictureBoxSizeMode::StretchImage;
			this->pbox_output->TabIndex = 5;
			this->pbox_output->TabStop = false;
			this->pbox_output->Click += gcnew System::EventHandler(this, &Pavic_gui_2024_Form::pbox_output_Click);
			// 
			// bt_copy
			// 
			this->bt_copy->Location = System::Drawing::Point(9, 51);
			this->bt_copy->Margin = System::Windows::Forms::Padding(2);
			this->bt_copy->Name = L"bt_copy";
			this->bt_copy->Size = System::Drawing::Size(142, 37);
			this->bt_copy->TabIndex = 6;
			this->bt_copy->Text = L"Copy";
			this->bt_copy->UseVisualStyleBackColor = true;
			this->bt_copy->Click += gcnew System::EventHandler(this, &Pavic_gui_2024_Form::bt_copy_Click);
			// 
			// bt_filter_bw
			// 
			this->bt_filter_bw->BackColor = System::Drawing::Color::White;
			this->bt_filter_bw->ForeColor = System::Drawing::Color::Black;
			this->bt_filter_bw->Location = System::Drawing::Point(216, 11);
		 this->bt_filter_bw->Margin = System::Windows::Forms::Padding(2);
			this->bt_filter_bw->Name = L"bt_filter_bw";
			this->bt_filter_bw->Size = System::Drawing::Size(142, 37);
			this->bt_filter_bw->TabIndex = 7;
			this->bt_filter_bw->Text = L"BW - CPU";
			this->bt_filter_bw->UseVisualStyleBackColor = true;
			this->bt_filter_bw->Click += gcnew System::EventHandler(this, &Pavic_gui_2024_Form::bt_filter_bw_Click);
			// 
			// label1
			// 
			this->label1->AutoSize = true;
			this->label1->Location = System::Drawing::Point(1078, 576);
			this->label1->Margin = System::Windows::Forms::Padding(2, 0, 2, 0);
			this->label1->Name = L"label1";
			this->label1->Size = System::Drawing::Size(128, 13);
			this->label1->TabIndex = 8;
			this->label1->Text = L" Ian Oliveira Teixeira";
			this->label1->ForeColor = System::Drawing::Color::White;
			// 
			// label2
			// 
			this->label2->AutoSize = true;
			this->label2->Location = System::Drawing::Point(7, 576);
			this->label2->Margin = System::Windows::Forms::Padding(2, 0, 2, 0);
			this->label2->Name = L"label2";
			this->label2->Size = System::Drawing::Size(94, 13);
			this->label2->TabIndex = 9;
			this->label2->Text = L" PAVIC LAB: 2025";
			this->label2->ForeColor = System::Drawing::Color::White;
			// 
			// bt_close_output
			// 
			this->bt_close_output->Location = System::Drawing::Point(734, 192);
			this->bt_close_output->Margin = System::Windows::Forms::Padding(2);
			this->bt_close_output->Name = L"bt_close_output";
			this->bt_close_output->Size = System::Drawing::Size(95, 28);
			this->bt_close_output->TabIndex = 11;
			this->bt_close_output->Text = L"Close";
			this->bt_close_output->UseVisualStyleBackColor = true;
			this->bt_close_output->Click += gcnew System::EventHandler(this, &Pavic_gui_2024_Form::bt_close_output_Click);
			// 
			// bt_filter_Sepia
			// 
			this->bt_filter_Sepia->BackColor = System::Drawing::Color::White;
			this->bt_filter_Sepia->ForeColor = System::Drawing::Color::Black;
			this->bt_filter_Sepia->Location = System::Drawing::Point(216, 93);
			this->bt_filter_Sepia->Margin = System::Windows::Forms::Padding(2);
			this->bt_filter_Sepia->Name = L"bt_filter_Sepia";
			this->bt_filter_Sepia->Size = System::Drawing::Size(142, 37);
			this->bt_filter_Sepia->TabIndex = 12;
			this->bt_filter_Sepia->Text = L"Sepia - CPU";
			this->bt_filter_Sepia->UseVisualStyleBackColor = true;
			this->bt_filter_Sepia->Click += gcnew System::EventHandler(this, &Pavic_gui_2024_Form::bt_filter_Sepia_Click);
			// 
			// bt_filter_Sepia_MultiThread
			// 
			this->bt_filter_Sepia_MultiThread->BackColor = System::Drawing::Color::White;
			this->bt_filter_Sepia_MultiThread->ForeColor = System::Drawing::Color::Black;
			this->bt_filter_Sepia_MultiThread->Location = System::Drawing::Point(216, 51);
			this->bt_filter_Sepia_MultiThread->Margin = System::Windows::Forms::Padding(2);
			this->bt_filter_Sepia_MultiThread->Name = L"bt_filter_Sepia_MultiThread";
			this->bt_filter_Sepia_MultiThread->Size = System::Drawing::Size(142, 37);
			this->bt_filter_Sepia_MultiThread->TabIndex = 13;
			this->bt_filter_Sepia_MultiThread->Text = L"BW - CUDA";
			this->bt_filter_Sepia_MultiThread->UseVisualStyleBackColor = true;
			this->bt_filter_Sepia_MultiThread->Click += gcnew System::EventHandler(this, &Pavic_gui_2024_Form::bt_filter_Sepia_MultiThread_Click);
			// 
			// bt_filter_Sepia_top
			// 
			this->bt_filter_Sepia_top->BackColor = System::Drawing::Color::White;
			this->bt_filter_Sepia_top->ForeColor = System::Drawing::Color::Black;
			this->bt_filter_Sepia_top->Location = System::Drawing::Point(362, 53);
			this->bt_filter_Sepia_top->Margin = System::Windows::Forms::Padding(2);
			this->bt_filter_Sepia_top->Name = L"bt_filter_Sepia_top";
			this->bt_filter_Sepia_top->Size = System::Drawing::Size(142, 37);
			this->bt_filter_Sepia_top->TabIndex = 14;
			this->bt_filter_Sepia_top->Text = L"Invert - CPU";
			this->bt_filter_Sepia_top->UseVisualStyleBackColor = true;
			this->bt_filter_Sepia_top->Click += gcnew System::EventHandler(this, &Pavic_gui_2024_Form::bt_filter_Sepia_top_Click);
			// 
			// lb_timer
			// 
			this->lb_timer->AutoSize = true;
			this->lb_timer->ForeColor = System::Drawing::Color::Red;
			this->lb_timer->Location = System::Drawing::Point(900, 20);
			this->lb_timer->Name = L"lb_timer";
			this->lb_timer->Size = System::Drawing::Size(79, 13);
			this->lb_timer->TabIndex = 16;
			this->lb_timer->Text = L"Execution time:";
			this->lb_timer->Click += gcnew System::EventHandler(this, &Pavic_gui_2024_Form::lb_timer_Click);
			// 
			// textB_Time
			// 
			this->textB_Time->BackColor = System::Drawing::Color::FromArgb(static_cast<System::Int32>(static_cast<System::Byte>(4)), static_cast<System::Int32>(static_cast<System::Byte>(8)),
				static_cast<System::Int32>(static_cast<System::Byte>(52)));
			this->textB_Time->ForeColor = System::Drawing::Color::Red;
			this->textB_Time->Location = System::Drawing::Point(900, 40);
			this->textB_Time->Name = L"textB_Time";
			this->textB_Time->ReadOnly = true;
			this->textB_Time->Size = System::Drawing::Size(170, 20);
			this->textB_Time->TabIndex = 17;
			
			// lbCudaStatus
			/*this->lbCudaStatus = (gcnew System::Windows::Forms::Label());
			this->lbCudaStatus->AutoSize = true;
			this->lbCudaStatus->Location = System::Drawing::Point(900, 70);
			this->lbCudaStatus->Name = L"lbCudaStatus";
			this->lbCudaStatus->Size = System::Drawing::Size(100, 13);
			this->lbCudaStatus->TabIndex = 18;
			this->lbCudaStatus->Text = L"CUDA: checking...";
			this->lbCudaStatus->ForeColor = System::Drawing::Color::White;*/
			// 
			// btCuda
			// 
			this->btCuda->BackColor = System::Drawing::Color::White;
			this->btCuda->ForeColor = System::Drawing::Color::Black;
			this->btCuda->Location = System::Drawing::Point(362, 12);
			this->btCuda->Margin = System::Windows::Forms::Padding(2);
			this->btCuda->Name = L"btCuda";
			this->btCuda->Size = System::Drawing::Size(142, 37);
			this->btCuda->TabIndex = 21;
			this->btCuda->Text = L"Sepia - CUDA";
			this->btCuda->UseVisualStyleBackColor = true;
			this->btCuda->Click += gcnew System::EventHandler(this, &Pavic_gui_2024_Form::btCuda_Click);
			// 
			// btInvertCuda
			// 
			this->btInvertCuda->BackColor = System::Drawing::Color::White;
			this->btInvertCuda->ForeColor = System::Drawing::Color::Black;
			this->btInvertCuda->Location = System::Drawing::Point(363, 93);
			this->btInvertCuda->Name = L"btInvertCuda";
			this->btInvertCuda->Size = System::Drawing::Size(142, 37);
			this->btInvertCuda->TabIndex = 22;
			this->btInvertCuda->Text = L"Invert - CUDA";
			this->btInvertCuda->UseVisualStyleBackColor = true;
			this->btInvertCuda->Click += gcnew System::EventHandler(this, &Pavic_gui_2024_Form::btInvertCuda_Click);
			// 
			// Pavic_gui_2024_Form
			// 
			this->AutoScaleDimensions = System::Drawing::SizeF(6, 13);
			this->AutoScaleMode = System::Windows::Forms::AutoScaleMode::Font;
			this->BackColor = System::Drawing::Color::FromArgb(static_cast<System::Int32>(static_cast<System::Byte>(4)), static_cast<System::Int32>(static_cast<System::Byte>(8)),
				static_cast<System::Int32>(static_cast<System::Byte>(52)));
			this->ClientSize = System::Drawing::Size(1251, 617);
			this->Controls->Add(this->btInvertCuda);
			this->Controls->Add(this->btCuda);
			this->Controls->Add(this->textB_Time);
			this->Controls->Add(this->lb_timer);
			this->Controls->Add(this->bt_filter_Sepia_top);
			this->Controls->Add(this->bt_filter_Sepia_MultiThread);
			this->Controls->Add(this->bt_filter_Sepia);
			this->Controls->Add(this->bt_close_output);
			this->Controls->Add(this->label2);
			this->Controls->Add(this->label1);
			this->Controls->Add(this->bt_filter_bw);
			this->Controls->Add(this->bt_copy);
			this->Controls->Add(this->pbox_output);
			this->Controls->Add(this->pbox_input);
			this->Controls->Add(this->bt_exit);
			this->Controls->Add(this->bt_close);
			this->Controls->Add(this->bt_open);
			this->Margin = System::Windows::Forms::Padding(2);
			this->Name = L"Pavic_gui_2024_Form";
			this->Text = L"PROJECT: PAVIC LAB 2025";
			this->Load += gcnew System::EventHandler(this, &Pavic_gui_2024_Form::Pavic_gui_2024_Form_Load);
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->pbox_input))->EndInit();
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->pbox_output))->EndInit();
			this->ResumeLayout(false);
			this->PerformLayout();

		}
#pragma endregion
	private: System::Void bt_open_Click(System::Object^ sender, System::EventArgs^ e) {
		OpenFileDialog^ ofd = gcnew OpenFileDialog();
		if (ofd->ShowDialog() == System::Windows::Forms::DialogResult::OK) {
			pbox_input->ImageLocation = ofd->FileName;
		}
	}
private: System::Void bt_close_Click(System::Object^ sender, System::EventArgs^ e) {

	pbox_input->Image = nullptr;

	
}
private: System::Void bt_copy_Click(System::Object^ sender, System::EventArgs^ e) {
	auto start = high_resolution_clock::now();
	pbox_output->Image = pbox_input->Image;
	auto end = high_resolution_clock::now();
	auto duration = duration_cast<milliseconds>(end - start);
	lb_timer->Text = "Tempo de cópia: " + duration.count().ToString() + " ms";
	textB_Time->Text = "PAVIC lab 2025";
}
private: System::Void bt_filter_bw_Click(System::Object^ sender, System::EventArgs^ e) {
	if (pbox_input->Image == nullptr) {
		MessageBox::Show("Please open an image first.", "No Image Loaded", MessageBoxButtons::OK, MessageBoxIcon::Warning);
		return;
	}
	Bitmap^ inputImage = (Bitmap^)pbox_input->Image;
	auto start = high_resolution_clock::now();
	Bitmap^ outputImage = nullptr;
	try {
		outputImage = processor->ApplyBW_CPU(inputImage);
	}
	catch (Exception^ ex) {
		MessageBox::Show(ex->Message, "Error", MessageBoxButtons::OK, MessageBoxIcon::Error);
		return;
	}
	auto end = high_resolution_clock::now();
	auto duration = duration_cast<milliseconds>(end - start);
	pbox_output->Image = outputImage;
	textB_Time->Text = duration.count().ToString() + " ms";
}
private: System::Void bt_exit_Click(System::Object^ sender, System::EventArgs^ e) {
	Application::Exit();
}
private: System::Void bt_close_copy_Click(System::Object^ sender, System::EventArgs^ e) {
	pbox_output->Image = nullptr;
	
}
private: System::Void bt_close_output_Click(System::Object^ sender, System::EventArgs^ e) {
	pbox_output->Image = nullptr;
}
private: System::Void Pavic_gui_2024_Form_Load(System::Object^ sender, System::EventArgs^ e) {
			// No CUDA status UI
}
private: System::Void pbox_input_Click(System::Object^ sender, System::EventArgs^ e) {
}
private: System::Void pbox_output_Click(System::Object^ sender, System::EventArgs^ e) {
    // Intentionally left empty
}
private: System::Void lb_timer_Click(System::Object^ sender, System::EventArgs^ e) {
    // Intentionally left empty
}
private: System::Void bt_filter_Sepia_Click(System::Object^ sender, System::EventArgs^ e) {
	if (pbox_input->Image == nullptr) {
		MessageBox::Show("Please open an image first.", "No Image Loaded", MessageBoxButtons::OK, MessageBoxIcon::Warning);
		return;
	}
	Bitmap^ inputImage = (Bitmap^)pbox_input->Image;
	auto start = high_resolution_clock::now();
	Bitmap^ outputImage = nullptr;
	try {
		outputImage = processor->ApplySepia_CPU(inputImage);
	}
	catch (Exception^ ex) {
		MessageBox::Show(ex->Message, "Error", MessageBoxButtons::OK, MessageBoxIcon::Error);
		return;
	}
	auto end = high_resolution_clock::now();
	auto duration = duration_cast<milliseconds>(end - start);
	pbox_output->Image = outputImage;
	textB_Time->Text = duration.count().ToString() + " ms";
}
private: System::Void bt_filter_Sepia_MultiThread_Click(System::Object^ sender, System::EventArgs^ e) {
	// This button is now BW - CUDA
	if (pbox_input->Image == nullptr) {
		MessageBox::Show("Please open an image first.", "No Image Loaded", MessageBoxButtons::OK, MessageBoxIcon::Warning);
		return;
	}
	Bitmap^ inputImage = (Bitmap^)pbox_input->Image;
	auto start = high_resolution_clock::now();
	Bitmap^ outputImage = nullptr;
	try {
		outputImage = processor->ApplyBW_CUDA(inputImage);
	}
	catch (Exception^ ex) {
		MessageBox::Show(ex->Message, "CUDA Error", MessageBoxButtons::OK, MessageBoxIcon::Error);
		return;
	}
	auto end = high_resolution_clock::now();
	auto duration = duration_cast<milliseconds>(end - start);
	pbox_output->Image = outputImage;
	textB_Time->Text = duration.count().ToString() + " ms";
}
private: System::Void btCuda_Click(System::Object^ sender, System::EventArgs^ e) {
	if (pbox_input->Image == nullptr) {
		MessageBox::Show("Please open an image first.", "No Image Loaded", MessageBoxButtons::OK, MessageBoxIcon::Warning);
		return;
	}
	Bitmap^ inputImage = (Bitmap^)pbox_input->Image;
	auto start = high_resolution_clock::now();
	Bitmap^ outputImage = nullptr;
	try {
		outputImage = processor->ApplySepia_CUDA(inputImage);
	}
	catch (Exception^ ex) {
		MessageBox::Show(ex->Message, "CUDA Error", MessageBoxButtons::OK, MessageBoxIcon::Error);
		return;
	}
	auto end = high_resolution_clock::now();
	auto duration = duration_cast<milliseconds>(end - start);
	pbox_output->Image = outputImage;
	textB_Time->Text = duration.count().ToString() + " ms";
}
private: System::Void btInvertCuda_Click(System::Object^ sender, System::EventArgs^ e) {
	if (pbox_input->Image == nullptr) {
		MessageBox::Show("Please open an image first.", "No Image Loaded", MessageBoxButtons::OK, MessageBoxIcon::Warning);
		return;
	}
	Bitmap^ inputImage = (Bitmap^)pbox_input->Image;
	auto start = high_resolution_clock::now();
	Bitmap^ outputImage = nullptr;
	try {
		outputImage = processor->ApplyInvert_CUDA(inputImage);
	}
	catch (Exception^ ex) {
		MessageBox::Show(ex->Message, "CUDA Error", MessageBoxButtons::OK, MessageBoxIcon::Error);
		return;
	}
	auto end = high_resolution_clock::now();
	auto duration = duration_cast<milliseconds>(end - start);
	pbox_output->Image = outputImage;
	textB_Time->Text = duration.count().ToString() + " ms";
}
private: System::Void bt_filter_Sepia_top_Click(System::Object^ sender, System::EventArgs^ e) {
	// Invert - CPU
	if (pbox_input->Image == nullptr) {
		MessageBox::Show("Please open an image first.", "No Image Loaded", MessageBoxButtons::OK, MessageBoxIcon::Warning);
		return;
	}
	Bitmap^ inputImage = (Bitmap^)pbox_input->Image;
	auto start = high_resolution_clock::now();
	Bitmap^ outputImage = nullptr;
	try {
		outputImage = processor->ApplyInvert_CPU(inputImage);
	}
	catch (Exception^ ex) {
		MessageBox::Show(ex->Message, "Error", MessageBoxButtons::OK, MessageBoxIcon::Error);
		return;
	}
	auto end = high_resolution_clock::now();
	auto duration = duration_cast<milliseconds>(end - start);
	pbox_output->Image = outputImage;
	textB_Time->Text = duration.count().ToString() + " ms";
}
private: System::Void bt_filter_Sepia_left_Click_1(System::Object^ sender, System::EventArgs^ e) {
	if (pbox_input->Image == nullptr) {
		MessageBox::Show("Please open an image first.", "No Image Loaded", MessageBoxButtons::OK, MessageBoxIcon::Warning);
		return;
	}
	Bitmap^ inputImage = (Bitmap^)pbox_input->Image;
	Bitmap^ outputImage = gcnew Bitmap(inputImage->Width, inputImage->Height);
	auto start = high_resolution_clock::now();
	for (int i = 0; i < inputImage->Width / 2; i++) {
		for (int j = 0; j < inputImage->Height; j++) {
			Color pixelColor = inputImage->GetPixel(i, j);
			int r = pixelColor.R;
			int g = pixelColor.G;
			int b = pixelColor.B;
			double tr = 0.393 * r + 0.769 * g + 0.189 * b;
			double tg = 0.349 * r + 0.686 * g + 0.168 * b;
			double tb = 0.272 * r + 0.534 * g + 0.131 * b;
			int newR = Math::Min(255, (int)tr);
			int newG = Math::Min(255, (int)tg);
			int newB = Math::Min(255, (int)tb);
			outputImage->SetPixel(i, j, Color::FromArgb(newR, newG, newB));
		}
	}
	auto end = high_resolution_clock::now();
	auto duration = duration_cast<milliseconds>(end - start);
	pbox_output->Image = outputImage;
	lb_timer->Text = "Sepia Left CPU duration: " + duration.count().ToString() + " ms";
}
private: System::Void button1_Click_1(System::Object^ sender, System::EventArgs^ e) {
	if (pbox_input->Image == nullptr) {
		MessageBox::Show("Please open an image first.", "No Image Loaded", MessageBoxButtons::OK, MessageBoxIcon::Warning);
		return;
	}
	Bitmap^ inputImage = (Bitmap^)pbox_input->Image;
	Bitmap^ outputImage = gcnew Bitmap(inputImage->Width, inputImage->Height);
	auto start = high_resolution_clock::now();
	for (int i = inputImage->Width / 2; i < inputImage->Width; i++) {
		for (int j = 0; j < inputImage->Height; j++) {
			Color pixelColor = inputImage->GetPixel(i, j);
			int r = pixelColor.R;
			int g = pixelColor.G;
			int b = pixelColor.B;
			double tr = 0.393 * r + 0.769 * g + 0.189 * b;
			double tg = 0.349 * r + 0.686 * g + 0.168 * b;
			double tb = 0.272 * r + 0.534 * g + 0.131 * b;
			int newR = Math::Min(255, (int)tr);
			int newG = Math::Min(255, (int)tg);
			int newB = Math::Min(255, (int)tb);
			outputImage->SetPixel(i, j, Color::FromArgb(newR, newG, newB));
		}
	}
	auto end = high_resolution_clock::now();
	auto duration = duration_cast<milliseconds>(end - start);
	pbox_output->Image = outputImage;
	lb_timer->Text = "Sepia Right CPU duration: " + duration.count().ToString() + " ms";
}
private: System::Void button2_Click(System::Object^ sender, System::EventArgs^ e) {
	if (pbox_input->Image == nullptr) {
		MessageBox::Show("Please open an image first.", "No Image Loaded", MessageBoxButtons::OK, MessageBoxIcon::Warning);
		return;
	}
	Bitmap^ inputImage = (Bitmap^)pbox_input->Image;
	Bitmap^ outputImage = gcnew Bitmap(inputImage->Width, inputImage->Height);
	auto start = high_resolution_clock::now();
	for (int i = 0; i < inputImage->Width; i++) {
		for (int j = inputImage->Height / 2; j < inputImage->Height; j++) {
			Color pixelColor = inputImage->GetPixel(i, j);
			int r = pixelColor.R;
			int g = pixelColor.G;
			int b = pixelColor.B;
			double tr = 0.393 * r + 0.769 * g + 0.189 * b;
			double tg = 0.349 * r + 0.686 * g + 0.168 * b;
			double tb = 0.272 * r + 0.534 * g + 0.131 * b;
			int newR = Math::Min(255, (int)tr);
			int newG = Math::Min(255, (int)tg);
			int newB = Math::Min(255, (int)tb);
			outputImage->SetPixel(i, j, Color::FromArgb(newR, newG, newB));
		}
	}
	auto end = high_resolution_clock::now();
	auto duration = duration_cast<milliseconds>(end - start);
	pbox_output->Image = outputImage;
	lb_timer->Text = "Sepia Bottom CPU duration: " + duration.count().ToString() + " ms";
}
private: System::Void bt_filter_Sepia_Thread_Click(System::Object^ sender, System::EventArgs^ e) {
	if (pbox_input->Image == nullptr) {
		MessageBox::Show("Please open an image first.", "No Image Loaded", MessageBoxButtons::OK, MessageBoxIcon::Warning);
		return;
	}
	Bitmap^ inputImage = (Bitmap^)pbox_input->Image;
	Bitmap^ outputImage = gcnew Bitmap(inputImage->Width, inputImage->Height);
	int height = inputImage->Height;
	auto start = high_resolution_clock::now();

	// Use unmanaged SepiaThreadArgs + unmanaged function to avoid storing managed handles inside std::thread internals.
	SepiaThreadArgs* argsTop = new SepiaThreadArgs();
	argsTop->inputImage = inputImage;
	argsTop->outputImage = outputImage;
	argsTop->startX = 0;
	argsTop->endX = inputImage->Width;
	argsTop->startY = 0;
	argsTop->endY = height / 2;

	SepiaThreadArgs* argsBottom = new SepiaThreadArgs();
	argsBottom->inputImage = inputImage;
	argsBottom->outputImage = outputImage;
	argsBottom->startX = 0;
	argsBottom->endX = inputImage->Width;
	argsBottom->startY = height / 2;
	argsBottom->endY = height;

	std::thread t1(ApplySepiaFilterWindow_Unmanaged, argsTop);
	std::thread t2(ApplySepiaFilterWindow_Unmanaged, argsBottom);

	t1.join();
	t2.join();

	auto end = high_resolution_clock::now();
	auto duration = duration_cast<milliseconds>(end - start);
	pbox_output->Image = outputImage;
	lb_timer->Text = "Sepia Thread duration: " + duration.count().ToString() + " ms";

	delete argsTop;
	delete argsBottom;
}

	}; // end class Pavic_gui_2024_Form

} // namespace pavicgui2024
